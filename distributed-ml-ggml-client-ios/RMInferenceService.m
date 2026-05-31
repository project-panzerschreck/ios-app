#import "RMInferenceService.h"
#import "RMChatMessage.h"
#import "RMRpcSettings.h"
#import "RMStorageServer.h"
#import <UIKit/UIKit.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <sys/sysctl.h>

NSString * const RMInferenceServiceDidUpdateNotification = @"RMInferenceServiceDidUpdateNotification";

@interface RMInferenceService ()

@property (nonatomic, assign) RMModelState modelState;
@property (nonatomic, assign) RMRPCServerState rpcServerState;
@property (nonatomic, copy) NSString *modelName;
@property (nonatomic, copy) NSString *modelErrorMessage;
@property (nonatomic, copy) NSString *rpcStatusMessage;
@property (nonatomic, copy) NSString *rpcEndpoint;
@property (nonatomic, assign) NSInteger modelLayers;
@property (nonatomic, assign) double tokensPerSecond;
@property (nonatomic, strong) LlamaModelInfo *modelInfo;
@property (nonatomic, copy) NSArray<RMChatMessage *> *chatMessages;
@property (nonatomic, strong) LlamaBridge *bridge;
@property (nonatomic) dispatch_queue_t workerQueue;
@property (nonatomic, assign) NSUInteger generationSequence;
@property (nonatomic, assign) NSUInteger rpcSequence;
@property (nonatomic, strong) RMStorageServer *storageServer;
@property (nonatomic) dispatch_source_t discoveryTimer;

@end

@implementation RMLocalInterface

- (instancetype)initWithInterfaceId:(NSString *)interfaceId label:(NSString *)label ip:(NSString *)ip {
    self = [super init];
    if (self) {
        _interfaceId = [interfaceId copy] ?: @"";
        _label = [label copy] ?: @"";
        _ip = [ip copy] ?: @"";
    }
    return self;
}

@end

@implementation RMInferenceService

+ (instancetype)sharedService {
    static RMInferenceService *service;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        service = [[RMInferenceService alloc] initPrivate];
    });
    return service;
}

- (instancetype)init {
    [NSException raise:@"Singleton" format:@"Use +sharedService"];
    return nil;
}

- (instancetype)initPrivate {
    self = [super init];
    if (self) {
        _bridge = [[LlamaBridge alloc] init];
        _workerQueue = dispatch_queue_create("rmcluster.inference.worker", DISPATCH_QUEUE_SERIAL);
        _modelState = RMModelStateUnloaded;
        _rpcServerState = RMRPCServerStateIdle;
        _chatMessages = @[];
        _modelName = @"";
        _modelErrorMessage = @"";
        _rpcStatusMessage = @"";
        _rpcEndpoint = @"";
    }
    return self;
}

- (void)postUpdate {
    [[NSNotificationCenter defaultCenter] postNotificationName:RMInferenceServiceDidUpdateNotification object:self];
}

- (void)notifyOnMain:(dispatch_block_t)block {
    dispatch_async(dispatch_get_main_queue(), ^{
        block();
        [self postUpdate];
    });
}

- (void)loadModelFromURL:(NSURL *)url contextLength:(NSInteger)contextLength {
    if (url == nil) {
        return;
    }
    self.generationSequence += 1;
    [self notifyOnMain:^{
        self.modelState = RMModelStateLoading;
        self.modelErrorMessage = @"";
        self.modelName = @"";
        self.modelLayers = 0;
        self.modelInfo = nil;
        self.tokensPerSecond = 0;
        self.chatMessages = @[];
    }];

    NSString *path = [url path];
    dispatch_async(self.workerQueue, ^{
        NSError *error = nil;
        BOOL success = [self.bridge loadModelFromPath:path nCtx:contextLength error:&error];
        LlamaModelInfo *info = self.bridge.modelInfo;
        [self notifyOnMain:^{
            if (success && info != nil) {
                self.modelState = RMModelStateReady;
                self.modelInfo = info;
                self.modelName = info.name ?: @"Unknown";
                self.modelLayers = info.nLayers;
                self.modelErrorMessage = @"";
            } else {
                self.modelState = RMModelStateError;
                self.modelErrorMessage = error.localizedDescription ?: @"Failed to load model.";
            }
        }];
    });
}

- (void)unloadModel {
    self.generationSequence += 1;
    dispatch_async(self.workerQueue, ^{
        [self.bridge unloadModel];
    });
    [self notifyOnMain:^{
        self.modelState = RMModelStateUnloaded;
        self.modelName = @"";
        self.modelLayers = 0;
        self.modelInfo = nil;
        self.modelErrorMessage = @"";
        self.tokensPerSecond = 0;
        self.chatMessages = @[];
    }];
}

- (void)sendMessage:(NSString *)text maxTokens:(NSInteger)maxTokens temperature:(float)temperature {
    NSString *trimmed = [[text ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    if (trimmed.length == 0 || self.modelState != RMModelStateReady) {
        return;
    }

    NSMutableArray<RMChatMessage *> *history = [self.chatMessages mutableCopy] ?: [NSMutableArray array];
    RMChatMessage *userMessage = [[RMChatMessage alloc] initWithRole:@"user" content:trimmed];
    RMChatMessage *assistantMessage = [[RMChatMessage alloc] initWithRole:@"assistant" content:@""];
    [history addObject:userMessage];
    [history addObject:assistantMessage];

    self.generationSequence += 1;
    NSUInteger sequence = self.generationSequence;

    [self notifyOnMain:^{
        self.modelState = RMModelStateGenerating;
        self.tokensPerSecond = 0;
        self.chatMessages = history;
        self.modelErrorMessage = @"";
    }];

    dispatch_async(self.workerQueue, ^{
        NSArray *messagePayload = [self dictionaryMessagesFromChatMessages:history];
        NSString *formatted = [self.bridge applyChatTemplate:messagePayload addAssistantPrefix:YES];
        if (formatted.length == 0) {
            [self notifyOnMain:^{
                if (sequence != self.generationSequence) {
                    return;
                }
                self.modelState = RMModelStateError;
                self.modelErrorMessage = @"Model has no chat template — cannot use conversation mode";
            }];
            return;
        }

        LlamaGenerationConfig *config = [LlamaGenerationConfig defaults];
        config.maxNewTokens = maxTokens;
        config.temperature = temperature;

        __block NSInteger tokenCount = 0;
        __block NSDate *startDate = [NSDate date];
        __block NSMutableString *accumulated = [NSMutableString string];

        [self.bridge generateFromPrompt:formatted config:config callback:^(NSString *token, BOOL done) {
            if (sequence != self.generationSequence) {
                return;
            }

            if (token.length > 0) {
                [accumulated appendString:token];
                tokenCount += 1;
            }

            NSTimeInterval elapsed = [[NSDate date] timeIntervalSinceDate:startDate];
            double tokensPerSecond = elapsed > 0 ? ((double)tokenCount / elapsed) : 0;

            [self notifyOnMain:^{
                if (sequence != self.generationSequence) {
                    return;
                }
                NSMutableArray<RMChatMessage *> *updated = [self.chatMessages mutableCopy] ?: [NSMutableArray array];
                if (updated.count > 0) {
                    RMChatMessage *lastMessage = [updated lastObject];
                    lastMessage.content = [accumulated copy];
                    self.chatMessages = updated;
                }
                self.tokensPerSecond = tokensPerSecond;
                if (done) {
                    self.modelInfo = self.bridge.modelInfo;
                    self.modelName = self.modelInfo.name ?: self.modelName;
                    self.modelLayers = self.modelInfo.nLayers;
                    self.modelState = self.modelInfo != nil ? RMModelStateReady : RMModelStateError;
                }
            }];
        }];
    });
}

- (NSArray *)dictionaryMessagesFromChatMessages:(NSArray<RMChatMessage *> *)messages {
    NSMutableArray *array = [NSMutableArray arrayWithCapacity:messages.count];
    for (RMChatMessage *message in messages) {
        [array addObject:@{
            @"role" : message.role ?: @"",
            @"content" : message.content ?: @""
        }];
    }
    return array;
}

- (void)cancelGeneration {
    self.generationSequence += 1;
    [self notifyOnMain:^{
        self.tokensPerSecond = 0;
        if (self.modelInfo != nil) {
            self.modelState = RMModelStateReady;
            self.modelName = self.modelInfo.name ?: self.modelName;
            self.modelLayers = self.modelInfo.nLayers;
        }
    }];
}

- (void)clearChat {
    self.generationSequence += 1;
    [self notifyOnMain:^{
        self.chatMessages = @[];
        self.tokensPerSecond = 0;
        if (self.modelInfo != nil) {
            self.modelState = RMModelStateReady;
        }
    }];
}

- (void)startRPCServerWithCoordinatorHost:(NSString *)coordinatorHost
                          coordinatorPort:(NSInteger)coordinatorPort
                                 nickname:(NSString *)nickname
                                  threads:(NSInteger)threads
                                 deviceId:(NSString *)deviceId {
    if (self.rpcServerState != RMRPCServerStateIdle) {
        return;
    }

    if (![LlamaBridge rpcAvailable]) {
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateUnavailable;
            self.rpcStatusMessage = @"ggml-rpc not compiled in. Run scripts/build-ggml-ios.sh then add ggml-rpc.xcframework to the target.";
        }];
        return;
    }

    self.rpcSequence += 1;
    NSUInteger sequence = self.rpcSequence;
    NSString *host = [RMRpcSettings listenHost];
    NSInteger port = [RMRpcSettings listenPort];
    NSInteger storagePort = [RMRpcSettings storagePort];
    self.storageServer = [[RMStorageServer alloc] initWithStorageDirectory:[[RMRpcSettings sharedSettings] storageDirectory]];
    [self.storageServer startOnPort:storagePort];

    [self startDiscoveryPingWithSequence:sequence
                         coordinatorHost:coordinatorHost
                         coordinatorPort:coordinatorPort
                                nickname:nickname
                             servicePort:port
                             storagePort:storagePort
                                deviceId:deviceId];

    [self notifyOnMain:^{
        self.rpcServerState = RMRPCServerStateStarting;
        self.rpcStatusMessage = @"Starting…";
        self.rpcEndpoint = @"";
        [UIApplication sharedApplication].idleTimerDisabled = YES;
    }];

    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(0.75 * NSEC_PER_SEC)), dispatch_get_main_queue(), ^{
        if (sequence != self.rpcSequence || self.rpcServerState != RMRPCServerStateStarting) {
            return;
        }
        self.rpcServerState = RMRPCServerStateRunning;
        self.rpcEndpoint = [NSString stringWithFormat:@"%@:%ld", host ?: @"0.0.0.0", (long)port];
        self.rpcStatusMessage = [NSString stringWithFormat:@"Listening on %@", self.rpcEndpoint];
        [self postUpdate];
    });

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSUInteger totalMB = [LlamaBridge processMemoryBudgetBytes] / 1048576ULL;
        NSUInteger freeMB = [LlamaBridge processAvailableMemoryBytes] / 1048576ULL;
        NSString *cacheDir = [[[NSFileManager defaultManager] URLsForDirectory:NSCachesDirectory inDomains:NSUserDomainMask].firstObject path];
        NSString *endpoint = [NSString stringWithFormat:@"%@:%ld", host ?: @"0.0.0.0", (long)port];
        [self.bridge startRPCServer:endpoint
                           cacheDir:cacheDir
                             freeMB:freeMB
                            totalMB:totalMB
                            threads:(NSUInteger)MAX(1, threads)];
        [self notifyOnMain:^{
            if (sequence != self.rpcSequence) {
                return;
            }
            [self stopDiscoveryPing];
            [self.storageServer stop];
            self.storageServer = nil;
            self.rpcServerState = RMRPCServerStateIdle;
            self.rpcStatusMessage = @"Stopped";
            self.rpcEndpoint = @"";
            [UIApplication sharedApplication].idleTimerDisabled = NO;
        }];
    });
}

- (void)stopRPCServer {
    self.rpcSequence += 1;
    [self stopDiscoveryPing];
    [self.storageServer stop];
    self.storageServer = nil;
    [self notifyOnMain:^{
        self.rpcServerState = RMRPCServerStateIdle;
        self.rpcStatusMessage = @"Stopped";
        self.rpcEndpoint = @"";
        [UIApplication sharedApplication].idleTimerDisabled = NO;
    }];
}

- (void)startDiscoveryPingWithSequence:(NSUInteger)sequence
                       coordinatorHost:(NSString *)coordinatorHost
                       coordinatorPort:(NSInteger)coordinatorPort
                              nickname:(NSString *)nickname
                           servicePort:(NSInteger)servicePort
                           storagePort:(NSInteger)storagePort
                              deviceId:(NSString *)deviceId {
    [self stopDiscoveryPing];
    NSString *trimmedHost = [[coordinatorHost ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    if (trimmedHost.length == 0) {
        return;
    }

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_UTILITY, 0);
    self.discoveryTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue);
    dispatch_source_set_timer(self.discoveryTimer, DISPATCH_TIME_NOW, (uint64_t)(10 * NSEC_PER_SEC), (uint64_t)(1 * NSEC_PER_SEC));
    __weak typeof(self) weakSelf = self;
    dispatch_source_set_event_handler(self.discoveryTimer, ^{
        typeof(self) strongSelf = weakSelf;
        if (strongSelf == nil || sequence != strongSelf.rpcSequence) {
            return;
        }
        [strongSelf sendDiscoveryAnnouncementToHost:trimmedHost
                                               port:coordinatorPort
                                          nickname:nickname
                                        servicePort:servicePort
                                        storagePort:storagePort
                                           deviceId:deviceId];
    });
    dispatch_resume(self.discoveryTimer);
}

- (void)stopDiscoveryPing {
    if (self.discoveryTimer != nil) {
        dispatch_source_cancel(self.discoveryTimer);
        self.discoveryTimer = nil;
    }
}

- (void)sendDiscoveryAnnouncementToHost:(NSString *)host
                                   port:(NSInteger)port
                               nickname:(NSString *)nickname
                            servicePort:(NSInteger)servicePort
                            storagePort:(NSInteger)storagePort
                               deviceId:(NSString *)deviceId {
    UIDevice *device = [UIDevice currentDevice];
    device.batteryMonitoringEnabled = YES;
    NSString *ip = [[[self class] allLocalIPv4Interfaces].firstObject ip] ?: @"0.0.0.0";
    NSString *hardwareModel = [[self class] hardwareModel];

    NSURLComponents *components = [[NSURLComponents alloc] init];
    components.scheme = @"http";
    components.host = host;
    components.port = @(port);
    components.path = @"/announce";
    NSMutableArray<NSURLQueryItem *> *items = [NSMutableArray array];
    [items addObject:[NSURLQueryItem queryItemWithName:@"id" value:deviceId ?: @""]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"port" value:[NSString stringWithFormat:@"%ld", (long)servicePort]]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"storage_port" value:[NSString stringWithFormat:@"%ld", (long)storagePort]]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"ip" value:ip]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"model" value:hardwareModel]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"max_size"
                                                 value:[NSString stringWithFormat:@"%llu", (unsigned long long)[LlamaBridge processAvailableMemoryBytes]]]];
    NSString *trimmedNickname = [[nickname ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    if (trimmedNickname.length > 0) {
        [items addObject:[NSURLQueryItem queryItemWithName:@"nickname" value:trimmedNickname]];
    }
    if (device.batteryLevel >= 0.0f) {
        [items addObject:[NSURLQueryItem queryItemWithName:@"battery"
                                                     value:[NSString stringWithFormat:@"%.1f", device.batteryLevel * 100.0f]]];
    }
    components.queryItems = items;
    NSURL *url = components.URL;
    if (url == nil) {
        return;
    }

    NSURLRequest *request = [NSURLRequest requestWithURL:url cachePolicy:NSURLRequestReloadIgnoringLocalCacheData timeoutInterval:5.0];
    [[[NSURLSession sharedSession] dataTaskWithRequest:request] resume];
}

+ (NSArray<RMLocalInterface *> *)allLocalIPv4Interfaces {
    struct ifaddrs *ifAddr = NULL;
    if (getifaddrs(&ifAddr) != 0 || ifAddr == NULL) {
        return @[];
    }

    NSMutableArray<RMLocalInterface *> *results = [NSMutableArray array];
    NSMutableSet *seen = [NSMutableSet set];
    struct ifaddrs *cursor = ifAddr;
    while (cursor != NULL) {
        if (cursor->ifa_addr != NULL && cursor->ifa_addr->sa_family == AF_INET) {
            NSString *name = [NSString stringWithUTF8String:cursor->ifa_name];
            if (![name isEqualToString:@"lo0"] && ![seen containsObject:name]) {
                char host[NI_MAXHOST];
                getnameinfo(cursor->ifa_addr,
                            cursor->ifa_addr->sa_len,
                            host,
                            sizeof(host),
                            NULL,
                            0,
                            NI_NUMERICHOST);
                NSString *ip = [NSString stringWithUTF8String:host];
                [seen addObject:name];
                NSString *label = [self labelForInterface:name ip:ip];
                [results addObject:[[RMLocalInterface alloc] initWithInterfaceId:name label:label ip:ip]];
            }
        }
        cursor = cursor->ifa_next;
    }
    freeifaddrs(ifAddr);

    NSArray *order = @[ @"en0", @"bridge100", @"utun", @"pdp_ip0" ];
    return [results sortedArrayUsingComparator:^NSComparisonResult(RMLocalInterface *left, RMLocalInterface *right) {
        NSUInteger leftIndex = order.count;
        NSUInteger rightIndex = order.count;
        for (NSUInteger idx = 0; idx < order.count; idx += 1) {
            NSString *prefix = order[idx];
            if ([left.interfaceId hasPrefix:prefix] && leftIndex == order.count) {
                leftIndex = idx;
            }
            if ([right.interfaceId hasPrefix:prefix] && rightIndex == order.count) {
                rightIndex = idx;
            }
        }
        if (leftIndex == rightIndex) {
            return [left.interfaceId compare:right.interfaceId];
        }
        return leftIndex < rightIndex ? NSOrderedAscending : NSOrderedDescending;
    }];
}

+ (NSString *)labelForInterface:(NSString *)interfaceName ip:(NSString *)ip {
    if ([interfaceName isEqualToString:@"en0"]) {
        return @"Wi-Fi";
    }
    if ([interfaceName isEqualToString:@"bridge100"]) {
        return @"Hotspot";
    }
    if ([interfaceName hasPrefix:@"pdp_ip"]) {
        return @"Cellular";
    }
    if ([interfaceName hasPrefix:@"utun"]) {
        NSArray *parts = [ip componentsSeparatedByString:@"."];
        if (parts.count == 4) {
            NSInteger secondOctet = [parts[1] integerValue];
            if ([parts[0] integerValue] == 100 && secondOctet >= 64 && secondOctet <= 127) {
                return @"Tailscale";
            }
        }
        return @"VPN";
    }
    if ([interfaceName hasPrefix:@"en"]) {
        return @"Ethernet";
    }
    return interfaceName;
}

+ (NSString *)hardwareModel {
    size_t size = 0;
    sysctlbyname("hw.machine", NULL, &size, NULL, 0);
    if (size == 0) {
        return [UIDevice currentDevice].model ?: @"iPhone";
    }
    char *machine = calloc(1, size);
    sysctlbyname("hw.machine", machine, &size, NULL, 0);
    NSString *result = [NSString stringWithUTF8String:machine];
    free(machine);
    return result ?: [UIDevice currentDevice].model ?: @"iPhone";
}

@end
