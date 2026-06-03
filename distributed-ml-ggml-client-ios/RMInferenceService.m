#import "RMInferenceService.h"
#import "RMChatMessage.h"
#import "RMRpcSettings.h"
#import "RMStorageServer.h"
#import "Diagnostics/RMAppLogger.h"
#import <UIKit/UIKit.h>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/sysctl.h>

static const NSTimeInterval kRMStartupGraceInterval = 3.0;
static const NSTimeInterval kRMHealthCheckTimeout = 2.0;

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
@property (nonatomic) dispatch_source_t supervisorTimer;
@property (nonatomic, assign) BOOL nodeShouldBeRunning;
@property (nonatomic, assign) BOOL appIsActive;
@property (nonatomic, assign) BOOL runtimeIsShuttingDown;
@property (nonatomic, assign) BOOL rpcWorkerActive;
@property (nonatomic, assign) BOOL rpcHealthy;
@property (nonatomic, assign) BOOL storageHealthy;
@property (nonatomic, copy) NSString *currentRPCEndpoint;
@property (nonatomic, copy) NSString *lastRuntimeError;
@property (nonatomic, copy) NSString *desiredCoordinatorHost;
@property (nonatomic, assign) NSInteger desiredCoordinatorPort;
@property (nonatomic, copy) NSString *desiredNickname;
@property (nonatomic, assign) NSInteger desiredThreads;
@property (nonatomic, copy) NSString *desiredDeviceId;
@property (nonatomic, strong) NSDate *lastRPCStartAt;
@property (nonatomic, strong) NSDate *lastStorageStartAt;

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
        _currentRPCEndpoint = @"";
        _lastRuntimeError = @"";
        _appIsActive = YES;
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
    NSString *trimmedHost = [[coordinatorHost ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    if (trimmedHost.length == 0) {
        [RMAppLogger logWithLevel:@"WARN" tag:@"GENERAL" message:@"node.start.rejected reason=missing_coordinator_host"];
        return;
    }

    if (![LlamaBridge rpcAvailable]) {
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateUnavailable;
            self.rpcStatusMessage = @"ggml-rpc not compiled in. Rebuild the iOS XCFrameworks with GGML_RPC=ON from the local llama.cpp-rpc checkout.";
        }];
        return;
    }

    self.desiredCoordinatorHost = trimmedHost;
    self.desiredCoordinatorPort = coordinatorPort;
    self.desiredNickname = nickname ?: @"";
    self.desiredThreads = MAX(1, threads);
    self.desiredDeviceId = deviceId ?: @"";
    self.nodeShouldBeRunning = YES;
    self.runtimeIsShuttingDown = NO;
    self.lastRuntimeError = @"";
    self.rpcSequence += 1;

    [RMAppLogger logWithLevel:@"INFO" tag:@"GENERAL" message:[NSString stringWithFormat:@"node.start.requested coordinator=%@:%ld", trimmedHost, (long)coordinatorPort]];
    [self notifyOnMain:^{
        self.rpcServerState = RMRPCServerStateStarting;
        self.rpcStatusMessage = @"Starting…";
        self.rpcEndpoint = @"";
    }];
    [self applyKeepAwakePolicy];
    [self publishRuntimeHealthWithStatus:@"starting"];

    if (self.appIsActive) {
        [self startNodeSupervisorIfNeeded];
    } else {
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateDegraded;
            self.rpcStatusMessage = @"Waiting for app to become active";
        }];
        [self publishRuntimeHealthWithStatus:@"degraded"];
    }
}

- (void)stopRPCServer {
    [RMAppLogger logWithLevel:@"INFO" tag:@"GENERAL" message:@"node.stop.requested reason=user"];
    [self stopNodeRuntimePreservingDesiredConfig:NO reason:@"Stopped by user"];
}

- (void)handleAppDidBecomeActive {
    self.appIsActive = YES;
    [RMAppLogger logWithLevel:@"INFO" tag:@"GENERAL" message:@"app.active"];
    [self applyKeepAwakePolicy];
    if (self.nodeShouldBeRunning) {
        if (self.rpcServerState == RMRPCServerStateDegraded) {
            [self notifyOnMain:^{
                self.rpcServerState = RMRPCServerStateStarting;
                self.rpcStatusMessage = @"Starting…";
            }];
        }
        [self startNodeSupervisorIfNeeded];
    } else {
        [self publishRuntimeHealthWithStatus:@"idle"];
    }
}

- (void)handleAppWillResignActive {
    self.appIsActive = NO;
    [RMAppLogger logWithLevel:@"INFO" tag:@"GENERAL" message:@"app.inactive stopping_runtime=true"];
    if (self.nodeShouldBeRunning) {
        [self stopNodeRuntimePreservingDesiredConfig:YES reason:@"App moved to background"];
    } else {
        [self applyKeepAwakePolicy];
        [self publishRuntimeHealthWithStatus:@"idle"];
    }
}

- (void)stopNodeRuntimePreservingDesiredConfig:(BOOL)preserve reason:(NSString *)reason {
    self.runtimeIsShuttingDown = YES;
    self.rpcSequence += 1;
    [self stopDiscoveryPing];
    [self stopNodeSupervisor];
    [self stopRPCWorkerWithReason:reason];
    [self stopStorageServer];

    if (preserve) {
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateDegraded;
            self.rpcStatusMessage = reason ?: @"Degraded";
            self.rpcEndpoint = @"";
        }];
    } else {
        self.nodeShouldBeRunning = NO;
        self.lastRuntimeError = @"";
        self.currentRPCEndpoint = @"";
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateIdle;
            self.rpcStatusMessage = reason ?: @"Stopped";
            self.rpcEndpoint = @"";
        }];
    }

    self.rpcHealthy = NO;
    self.storageHealthy = NO;
    [self applyKeepAwakePolicy];
    [self publishRuntimeHealthWithStatus:preserve ? @"degraded" : @"idle"];
}

- (void)startNodeSupervisorIfNeeded {
    if (self.supervisorTimer != nil || !self.nodeShouldBeRunning || !self.appIsActive) {
        return;
    }
    [RMAppLogger logWithLevel:@"INFO" tag:@"GENERAL" message:@"supervisor.start"];
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_UTILITY, 0);
    self.supervisorTimer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, queue);
    dispatch_source_set_timer(self.supervisorTimer, DISPATCH_TIME_NOW, (uint64_t)(1.0 * NSEC_PER_SEC), (uint64_t)(0.25 * NSEC_PER_SEC));
    __weak typeof(self) weakSelf = self;
    dispatch_source_set_event_handler(self.supervisorTimer, ^{
        [weakSelf runNodeSupervisorIteration];
    });
    dispatch_resume(self.supervisorTimer);
}

- (void)stopNodeSupervisor {
    if (self.supervisorTimer != nil) {
        dispatch_source_cancel(self.supervisorTimer);
        self.supervisorTimer = nil;
    }
}

- (void)runNodeSupervisorIteration {
    if (!self.nodeShouldBeRunning || !self.appIsActive) {
        [self publishRuntimeHealthWithStatus:[self currentRuntimeStatusName]];
        return;
    }

    [self applyKeepAwakePolicy];
    [self startStorageServerIfNeeded];

    if (!self.rpcWorkerActive) {
        [self startRPCWorker];
    }

    BOOL rpcProbe = [self probeTCPOnHost:@"127.0.0.1" port:[RMRpcSettings listenPort]];
    BOOL storageProbe = [self probeStorageHealth];
    BOOL rpcWithinGrace = self.lastRPCStartAt != nil && [[NSDate date] timeIntervalSinceDate:self.lastRPCStartAt] < kRMStartupGraceInterval;
    BOOL storageWithinGrace = self.lastStorageStartAt != nil && [[NSDate date] timeIntervalSinceDate:self.lastStorageStartAt] < kRMStartupGraceInterval;
    BOOL effectiveRPCHealthy = rpcProbe || (self.rpcWorkerActive && rpcWithinGrace);
    BOOL effectiveStorageHealthy = storageProbe || (self.storageServer != nil && storageWithinGrace);

    self.rpcHealthy = effectiveRPCHealthy;
    self.storageHealthy = effectiveStorageHealthy;

    if (!effectiveStorageHealthy) {
        self.lastRuntimeError = @"Storage server unhealthy";
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateRecovering;
            self.rpcStatusMessage = self.lastRuntimeError;
        }];
        [self publishRuntimeHealthWithStatus:@"recovering"];
        if (!storageWithinGrace) {
            [RMAppLogger logWithLevel:@"WARN" tag:@"GENERAL" message:@"health.storage_unhealthy action=restart_storage"];
            [self restartStorageServer];
        }
        return;
    }

    if (!effectiveRPCHealthy) {
        self.lastRuntimeError = @"RPC worker unhealthy";
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateRecovering;
            self.rpcStatusMessage = self.lastRuntimeError;
        }];
        [self publishRuntimeHealthWithStatus:@"recovering"];
        if (!rpcWithinGrace) {
            [RMAppLogger logWithLevel:@"WARN" tag:@"GENERAL" message:@"health.rpc_unhealthy action=restart_rpc"];
            [self restartRPCWorkerWithReason:self.lastRuntimeError];
        }
        return;
    }

    NSString *endpoint = self.currentRPCEndpoint.length > 0 ? self.currentRPCEndpoint : [NSString stringWithFormat:@"%@:%ld", [RMRpcSettings listenHost], (long)[RMRpcSettings listenPort]];
    [self notifyOnMain:^{
        self.rpcServerState = RMRPCServerStateRunning;
        self.rpcEndpoint = endpoint;
        self.rpcStatusMessage = [NSString stringWithFormat:@"Listening on %@", endpoint];
        self.lastRuntimeError = @"";
    }];
    [self publishRuntimeHealthWithStatus:@"running"];
    [self sendDiscoveryAnnouncementToHost:self.desiredCoordinatorHost
                                       port:self.desiredCoordinatorPort
                                  nickname:self.desiredNickname
                               servicePort:[RMRpcSettings listenPort]
                               storagePort:[RMRpcSettings storagePort]
                                  deviceId:self.desiredDeviceId];
}

- (BOOL)startStorageServerIfNeeded {
    if (self.storageServer != nil) {
        return YES;
    }
    NSInteger storagePort = [RMRpcSettings storagePort];
    self.storageServer = [[RMStorageServer alloc] initWithStorageDirectory:[[RMRpcSettings sharedSettings] storageDirectory]];
    if (![self.storageServer startOnPort:storagePort]) {
        self.lastRuntimeError = [NSString stringWithFormat:@"Storage failed to bind port %ld", (long)storagePort];
        [RMAppLogger logWithLevel:@"ERROR" tag:@"GENERAL" message:[NSString stringWithFormat:@"storage.runtime.start_failed port=%ld", (long)storagePort]];
        return NO;
    }
    self.lastStorageStartAt = [NSDate date];
    [RMAppLogger logWithLevel:@"INFO" tag:@"STORAGE" message:[NSString stringWithFormat:@"storage.runtime.started port=%ld", (long)storagePort]];
    return YES;
}

- (void)restartStorageServer {
    [self stopStorageServer];
    [self startStorageServerIfNeeded];
}

- (void)stopStorageServer {
    [self.storageServer stop];
    self.storageServer = nil;
}

- (void)startRPCWorker {
    if (self.rpcWorkerActive) {
        return;
    }
    NSString *host = [RMRpcSettings listenHost];
    NSInteger port = [RMRpcSettings listenPort];
    self.currentRPCEndpoint = [NSString stringWithFormat:@"%@:%ld", host ?: @"0.0.0.0", (long)port];
    self.lastRPCStartAt = [NSDate date];
    self.rpcWorkerActive = YES;
    self.runtimeIsShuttingDown = NO;
    NSUInteger sequence = self.rpcSequence;

    [RMAppLogger logWithLevel:@"INFO" tag:@"RPC SERVER" message:[NSString stringWithFormat:@"rpc.start.begin endpoint=%@", self.currentRPCEndpoint]];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSUInteger totalMB = [LlamaBridge processMemoryBudgetBytes] / 1048576ULL;
        NSUInteger freeMB = [LlamaBridge processAvailableMemoryBytes] / 1048576ULL;
        NSString *cacheDir = [[[NSFileManager defaultManager] URLsForDirectory:NSCachesDirectory inDomains:NSUserDomainMask].firstObject path];
        [self.bridge startRPCServer:self.currentRPCEndpoint
                           cacheDir:cacheDir
                             freeMB:freeMB
                            totalMB:totalMB
                            threads:(NSUInteger)MAX(1, self.desiredThreads)];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (sequence != self.rpcSequence) {
                return;
            }
            self.rpcWorkerActive = NO;
            if (!self.runtimeIsShuttingDown && self.nodeShouldBeRunning && self.appIsActive) {
                self.lastRuntimeError = @"RPC worker exited unexpectedly";
                [RMAppLogger logWithLevel:@"ERROR" tag:@"RPC SERVER" message:@"rpc.exit.unexpected"];
                self.rpcServerState = RMRPCServerStateRecovering;
                self.rpcStatusMessage = self.lastRuntimeError;
                [self publishRuntimeHealthWithStatus:@"recovering"];
            }
            [self postUpdate];
        });
    });
}

- (void)stopRPCWorkerWithReason:(NSString *)reason {
    if (!self.rpcWorkerActive && self.currentRPCEndpoint.length == 0) {
        return;
    }
    [RMAppLogger logWithLevel:@"INFO" tag:@"RPC SERVER" message:[NSString stringWithFormat:@"rpc.stop.requested reason=%@", reason ?: @""]];
    self.runtimeIsShuttingDown = YES;
    if (self.currentRPCEndpoint.length > 0) {
        [self.bridge stopRPCServer:self.currentRPCEndpoint];
    }
    self.rpcWorkerActive = NO;
}

- (void)restartRPCWorkerWithReason:(NSString *)reason {
    [self stopRPCWorkerWithReason:reason];
    self.rpcWorkerActive = NO;
    [self startRPCWorker];
}

- (BOOL)probeTCPOnHost:(NSString *)host port:(NSInteger)port {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return NO;
    }
    struct timeval timeout;
    timeout.tv_sec = (time_t)kRMHealthCheckTimeout;
    timeout.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    inet_pton(AF_INET, host.UTF8String, &addr.sin_addr);

    BOOL connected = connect(sock, (struct sockaddr *)&addr, sizeof(addr)) == 0;
    close(sock);
    return connected;
}

- (BOOL)probeStorageHealth {
    if (self.storageServer == nil) {
        return NO;
    }
    NSURL *url = [NSURL URLWithString:[NSString stringWithFormat:@"http://127.0.0.1:%ld/storage_info", (long)[RMRpcSettings storagePort]]];
    if (url == nil) {
        return NO;
    }
    NSURLRequest *request = [NSURLRequest requestWithURL:url cachePolicy:NSURLRequestReloadIgnoringLocalCacheData timeoutInterval:kRMHealthCheckTimeout];
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    __block BOOL healthy = NO;
    [[[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error == nil) {
            NSHTTPURLResponse *http = (NSHTTPURLResponse *)response;
            healthy = http.statusCode == 200;
        }
        dispatch_semaphore_signal(sem);
    }] resume];
    dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)((kRMHealthCheckTimeout + 1.0) * NSEC_PER_SEC)));
    return healthy;
}

- (void)applyKeepAwakePolicy {
    [UIApplication sharedApplication].idleTimerDisabled = self.nodeShouldBeRunning && self.appIsActive;
}

- (void)publishRuntimeHealthWithStatus:(NSString *)status {
    [RMAppLogger rpcHealthWithStatus:status details:@{
        @"endpoint": self.currentRPCEndpoint ?: @"",
        @"coordinator": self.desiredCoordinatorHost.length > 0 ? [NSString stringWithFormat:@"%@:%ld", self.desiredCoordinatorHost, (long)self.desiredCoordinatorPort] : @"",
        @"last_error": self.lastRuntimeError ?: @"",
        @"rpc_healthy": @(self.rpcHealthy),
        @"storage_healthy": @(self.storageHealthy),
    }];
}

- (NSString *)currentRuntimeStatusName {
    switch (self.rpcServerState) {
        case RMRPCServerStateStarting:
            return @"starting";
        case RMRPCServerStateRunning:
            return @"running";
        case RMRPCServerStateRecovering:
            return @"recovering";
        case RMRPCServerStateDegraded:
            return @"degraded";
        case RMRPCServerStateUnavailable:
            return @"unavailable";
        default:
            return @"idle";
    }
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
