#import "RMInferenceService.h"
#import "RMChatMessage.h"
#import "RMRpcSettings.h"
#import "RMStorageServer.h"
#import "RMAppLogger.h"
#import <UIKit/UIKit.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <sys/socket.h>
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
@property (nonatomic, assign) BOOL nodeShouldBeRunning;
@property (nonatomic, assign) BOOL appIsActive;
@property (nonatomic, assign) BOOL runtimeIsShuttingDown;
@property (nonatomic, copy) NSString *desiredCoordinatorHost;
@property (nonatomic, assign) NSInteger desiredCoordinatorPort;
@property (nonatomic, copy) NSString *desiredNickname;
@property (nonatomic, assign) NSInteger desiredThreads;
@property (nonatomic, copy) NSString *desiredDeviceId;
@property (nonatomic, assign) NSUInteger supervisorGeneration;
@property (nonatomic, assign) BOOL rpcWorkerActive;
@property (nonatomic, copy) NSString *currentRPCEndpoint;
@property (nonatomic, copy) NSString *lastRuntimeError;
@property (nonatomic, strong) NSDate *lastRPCStartAt;
@property (nonatomic, strong) NSDate *lastStorageStartAt;
@property (nonatomic, assign) BOOL rpcHealthy;
@property (nonatomic, assign) BOOL storageHealthy;
@property (nonatomic, assign) BOOL announceEligible;
@property (nonatomic, assign) BOOL discoveryActive;
@property (nonatomic, strong) dispatch_queue_t supervisorQueue;

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
        _appIsActive = YES;
        _supervisorQueue = dispatch_queue_create("rmcluster.node.supervisor", DISPATCH_QUEUE_SERIAL);
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
    if (self.nodeShouldBeRunning) {
        return;
    }

    if (![LlamaBridge rpcAvailable]) {
        NSString *message = @"ggml-rpc not compiled in. Run scripts/build-ggml-ios10-armv7.sh then rebuild the app.";
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateUnavailable;
            self.rpcStatusMessage = message;
        }];
        [RMAppLogger logWithLevel:@"ERROR" tag:@"RPC SERVER" message:@"rpc.unavailable reason=not_compiled"];
        return;
    }

    self.desiredCoordinatorHost = [[coordinatorHost ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    self.desiredCoordinatorPort = coordinatorPort;
    self.desiredNickname = nickname ?: @"";
    self.desiredThreads = MAX(1, threads);
    self.desiredDeviceId = deviceId ?: @"";
    self.nodeShouldBeRunning = YES;
    self.lastRuntimeError = @"";

    [RMAppLogger logWithTag:@"GENERAL" message:[NSString stringWithFormat:@"node.start.requested coordinator=%@:%ld", self.desiredCoordinatorHost, (long)self.desiredCoordinatorPort]];
    [self applyKeepAwakePolicy];
    [self publishRuntimeHealthWithStatusOverride:@"starting"];

    if (self.appIsActive) {
        [self startNodeSupervisorIfNeeded];
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateStarting;
            self.rpcStatusMessage = @"Starting…";
            self.rpcEndpoint = @"";
        }];
    } else {
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateDegraded;
            self.rpcStatusMessage = @"Waiting for app to become active";
        }];
        [self publishRuntimeHealthWithStatusOverride:@"degraded"];
    }
}

- (void)stopRPCServer {
    [RMAppLogger logWithTag:@"GENERAL" message:@"node.stop.requested reason=user"];
    [self stopNodeRuntimePreservingDesiredConfig:NO reason:@"Stopped by user"];
}

- (void)handleAppDidBecomeActive {
    self.appIsActive = YES;
    [RMAppLogger logWithTag:@"GENERAL" message:@"app.active"];
    [self applyKeepAwakePolicy];
    if (self.nodeShouldBeRunning) {
        if (self.rpcServerState == RMRPCServerStateDegraded) {
            self.rpcServerState = RMRPCServerStateStarting;
        }
        [self startNodeSupervisorIfNeeded];
        [self postUpdate];
    } else {
        [self publishRuntimeHealthWithStatusOverride:@"idle"];
    }
}

- (void)handleAppWillResignActive {
    self.appIsActive = NO;
    [RMAppLogger logWithTag:@"GENERAL" message:@"app.inactive stopping_runtime=true"];
    if (self.nodeShouldBeRunning) {
        [self stopNodeRuntimePreservingDesiredConfig:YES reason:@"App moved to background"];
    } else {
        [self applyKeepAwakePolicy];
        [self publishRuntimeHealthWithStatusOverride:@"idle"];
    }
}

- (void)applyKeepAwakePolicy {
    dispatch_async(dispatch_get_main_queue(), ^{
        [UIApplication sharedApplication].idleTimerDisabled = self.nodeShouldBeRunning && self.appIsActive;
    });
}

- (void)startNodeSupervisorIfNeeded {
    if (!self.nodeShouldBeRunning || !self.appIsActive) {
        return;
    }
    self.supervisorGeneration += 1;
    NSUInteger generation = self.supervisorGeneration;
    [RMAppLogger logWithTag:@"GENERAL" message:@"supervisor.start"];
    [self scheduleSupervisorIterationAfter:0.1 generation:generation];
}

- (void)scheduleSupervisorIterationAfter:(NSTimeInterval)delay generation:(NSUInteger)generation {
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(delay * NSEC_PER_SEC)), self.supervisorQueue, ^{
        if (generation != self.supervisorGeneration || !self.nodeShouldBeRunning) {
            return;
        }
        NSTimeInterval nextDelay = [self runNodeSupervisorIteration];
        [self scheduleSupervisorIterationAfter:MAX(nextDelay, 0.5) generation:generation];
    });
}

- (void)stopNodeRuntimePreservingDesiredConfig:(BOOL)preserve reason:(NSString *)reason {
    self.runtimeIsShuttingDown = YES;
    self.supervisorGeneration += 1;
    self.discoveryActive = NO;
    [self stopStorageServer];
    [self stopRPCWorkerWithReason:reason];
    self.rpcHealthy = NO;
    self.storageHealthy = NO;
    self.announceEligible = NO;

    if (preserve) {
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateDegraded;
            self.rpcStatusMessage = reason ?: @"Degraded";
        }];
    } else {
        self.nodeShouldBeRunning = NO;
        self.desiredCoordinatorHost = @"";
        self.lastRuntimeError = @"";
        self.currentRPCEndpoint = @"";
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateIdle;
            self.rpcStatusMessage = @"Stopped";
            self.rpcEndpoint = @"";
        }];
    }

    [self applyKeepAwakePolicy];
    [self publishRuntimeHealthWithStatusOverride:preserve ? @"degraded" : @"idle"];
}

- (BOOL)startStorageServerIfNeeded {
    if (self.storageServer != nil) {
        return YES;
    }
    NSInteger storagePort = [RMRpcSettings storagePort];
    self.storageServer = [[RMStorageServer alloc] initWithStorageDirectory:[[RMRpcSettings sharedSettings] storageDirectory]];
    BOOL started = [self.storageServer startOnPort:storagePort];
    if (started) {
        self.lastStorageStartAt = [NSDate date];
        [RMAppLogger logWithTag:@"STORAGE" message:[NSString stringWithFormat:@"storage.runtime.started port=%ld", (long)storagePort]];
    } else {
        self.lastRuntimeError = [NSString stringWithFormat:@"Storage failed to bind port %ld", (long)storagePort];
        [RMAppLogger logWithLevel:@"ERROR" tag:@"GENERAL" message:self.lastRuntimeError];
    }
    return started;
}

- (void)stopStorageServer {
    [self.storageServer stop];
    self.storageServer = nil;
}

- (void)restartStorageServer {
    [RMAppLogger logWithTag:@"GENERAL" message:@"storage.runtime.restart_requested"];
    [self stopStorageServer];
    [self startStorageServerIfNeeded];
}

- (void)startRPCWorker {
    if (self.rpcWorkerActive) {
        return;
    }
    self.runtimeIsShuttingDown = NO;
    NSString *host = [RMRpcSettings listenHost];
    NSInteger port = [RMRpcSettings listenPort];
    self.currentRPCEndpoint = [NSString stringWithFormat:@"%@:%ld", host ?: @"0.0.0.0", (long)port];
    self.lastRPCStartAt = [NSDate date];
    self.rpcWorkerActive = YES;
    self.rpcSequence += 1;
    NSUInteger sequence = self.rpcSequence;

    NSUInteger totalMB = [NSProcessInfo processInfo].physicalMemory / 1048576ULL;
    NSUInteger rawFreeMB = [LlamaBridge processAvailableMemoryBytes] / 1048576ULL;
    NSUInteger freeMB = (NSUInteger)((double)rawFreeMB * 0.9);
    NSString *cacheDir = [[[NSFileManager defaultManager] URLsForDirectory:NSCachesDirectory inDomains:NSUserDomainMask].firstObject path];
    NSString *endpoint = [self.currentRPCEndpoint copy];
    NSInteger threads = self.desiredThreads;

    [RMAppLogger logWithTag:@"RPC SERVER" message:[NSString stringWithFormat:@"rpc.start.begin endpoint=%@ threads=%ld free_mb=%lu total_mb=%lu", endpoint, (long)threads, (unsigned long)freeMB, (unsigned long)totalMB]];

    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        [self.bridge startRPCServer:endpoint
                           cacheDir:cacheDir
                             freeMB:freeMB
                            totalMB:totalMB
                            threads:(NSUInteger)MAX(1, threads)];
        dispatch_async(dispatch_get_main_queue(), ^{
            if (sequence != self.rpcSequence) {
                return;
            }
            [self handleRPCWorkerExitWithEndpoint:endpoint];
        });
    });

    [self notifyOnMain:^{
        self.rpcEndpoint = endpoint;
        self.rpcServerState = RMRPCServerStateRunning;
        self.rpcStatusMessage = [NSString stringWithFormat:@"Listening on %@", endpoint];
    }];
}

- (void)stopRPCWorkerWithReason:(NSString *)reason {
    if (!self.rpcWorkerActive && self.currentRPCEndpoint.length == 0) {
        return;
    }
    [RMAppLogger logWithTag:@"RPC SERVER" message:[NSString stringWithFormat:@"rpc.stop.requested reason=%@", reason ?: @"unknown"]];
    self.runtimeIsShuttingDown = YES;
    if (self.currentRPCEndpoint.length > 0) {
        [self.bridge stopRPCServer:self.currentRPCEndpoint];
    }
    self.rpcSequence += 1;
    self.rpcWorkerActive = NO;
}

- (void)handleRPCWorkerExitWithEndpoint:(NSString *)endpoint {
    BOOL expected = self.runtimeIsShuttingDown || !self.nodeShouldBeRunning || !self.appIsActive;
    self.rpcWorkerActive = NO;
    self.rpcHealthy = NO;
    self.announceEligible = NO;

    if (expected) {
        [RMAppLogger logWithTag:@"RPC SERVER" message:[NSString stringWithFormat:@"rpc.exit.expected endpoint=%@", endpoint]];
        if (!self.nodeShouldBeRunning) {
            self.currentRPCEndpoint = @"";
            self.rpcServerState = RMRPCServerStateIdle;
            self.rpcEndpoint = @"";
        }
    } else {
        self.lastRuntimeError = @"RPC worker exited unexpectedly";
        self.rpcServerState = RMRPCServerStateRecovering;
        self.rpcStatusMessage = self.lastRuntimeError;
        [RMAppLogger logWithLevel:@"ERROR" tag:@"RPC SERVER" message:[NSString stringWithFormat:@"rpc.exit.unexpected endpoint=%@", endpoint]];
    }
    [self publishRuntimeHealthWithStatusOverride:expected ? [self currentRuntimeStatusName] : @"recovering"];
    [self postUpdate];
}

- (NSTimeInterval)runNodeSupervisorIteration {
    if (!self.nodeShouldBeRunning || !self.appIsActive) {
        [self publishRuntimeHealthWithStatusOverride:[self currentRuntimeStatusName]];
        return 10.0;
    }

    [self applyKeepAwakePolicy];
    if (![self startStorageServerIfNeeded]) {
        self.storageHealthy = NO;
        self.announceEligible = NO;
        self.discoveryActive = NO;
        self.lastRuntimeError = @"Storage server unhealthy";
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateRecovering;
            self.rpcStatusMessage = self.lastRuntimeError;
        }];
        [self publishRuntimeHealthWithStatusOverride:@"recovering"];
        [self restartStorageServer];
        return 2.0;
    }

    if (!self.rpcWorkerActive) {
        [self startRPCWorker];
    }

    BOOL rpcProbe = [self.class probeTCPHost:@"127.0.0.1" port:[RMRpcSettings listenPort] timeout:1.5];
    BOOL storageProbe = [self storageHealthProbe];
    BOOL rpcWithinGrace = self.lastRPCStartAt != nil && [[NSDate date] timeIntervalSinceDate:self.lastRPCStartAt] < 2.0;
    BOOL storageWithinGrace = self.lastStorageStartAt != nil && [[NSDate date] timeIntervalSinceDate:self.lastStorageStartAt] < 2.0;
    BOOL effectiveRPCHealthy = rpcProbe || (self.rpcWorkerActive && rpcWithinGrace);
    BOOL effectiveStorageHealthy = storageProbe || (self.storageServer != nil && storageWithinGrace);

    self.rpcHealthy = effectiveRPCHealthy;
    self.storageHealthy = effectiveStorageHealthy;
    self.announceEligible = self.rpcHealthy && self.storageHealthy;

    if (!self.storageHealthy) {
        self.discoveryActive = NO;
        self.lastRuntimeError = @"Storage server unhealthy";
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateRecovering;
            self.rpcStatusMessage = self.lastRuntimeError;
        }];
        [self publishRuntimeHealthWithStatusOverride:@"recovering"];
        if (!storageWithinGrace) {
            [RMAppLogger logWithLevel:@"WARN" tag:@"GENERAL" message:@"health.storage_unhealthy action=restart_storage announce=skipped"];
            [self restartStorageServer];
        }
        return 2.0;
    }

    if (!self.rpcHealthy) {
        self.discoveryActive = NO;
        self.lastRuntimeError = @"RPC worker unhealthy";
        [self notifyOnMain:^{
            self.rpcServerState = RMRPCServerStateRecovering;
            self.rpcStatusMessage = self.lastRuntimeError;
        }];
        [self publishRuntimeHealthWithStatusOverride:@"recovering"];
        if (!rpcWithinGrace) {
#if defined(GGML_RPC_NO_STOP_SERVER)
            [RMAppLogger logWithLevel:@"WARN" tag:@"GENERAL" message:@"health.rpc_unhealthy action=manual_restart_required announce=skipped"];
#else
            [RMAppLogger logWithLevel:@"WARN" tag:@"GENERAL" message:@"health.rpc_unhealthy action=restart_rpc announce=skipped"];
            [self stopRPCWorkerWithReason:self.lastRuntimeError];
            [self startRPCWorker];
#endif
        }
        return 2.0;
    }

    [self notifyOnMain:^{
        self.rpcServerState = RMRPCServerStateRunning;
        self.rpcStatusMessage = [NSString stringWithFormat:@"Listening on %@", self.currentRPCEndpoint ?: @""];
        self.rpcEndpoint = self.currentRPCEndpoint ?: @"";
    }];
    [self publishRuntimeHealthWithStatusOverride:@"running"];
    return [self announceToCoordinator];
}

- (BOOL)storageHealthProbe {
    if (self.storageServer == nil) {
        return NO;
    }
    NSURL *url = [NSURL URLWithString:[NSString stringWithFormat:@"http://127.0.0.1:%ld/storage_info", (long)[RMRpcSettings storagePort]]];
    if (url == nil) {
        return NO;
    }
    NSURLRequest *request = [NSURLRequest requestWithURL:url cachePolicy:NSURLRequestReloadIgnoringLocalCacheData timeoutInterval:1.5];
    __block BOOL healthy = NO;
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    [[[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error == nil) {
            NSHTTPURLResponse *http = (NSHTTPURLResponse *)response;
            healthy = http.statusCode == 200;
        }
        dispatch_semaphore_signal(sema);
    }] resume];
    dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(2.0 * NSEC_PER_SEC)));
    return healthy;
}

- (NSTimeInterval)announceToCoordinator {
    self.discoveryActive = YES;
    NSString *host = self.desiredCoordinatorHost;
    if (host.length == 0) {
        return 10.0;
    }

    UIDevice *device = [UIDevice currentDevice];
    device.batteryMonitoringEnabled = YES;
    NSString *ip = [[[self class] allLocalIPv4Interfaces].firstObject ip] ?: @"0.0.0.0";
    NSString *hardwareModel = [[self class] hardwareModel];
    NSInteger servicePort = [RMRpcSettings listenPort];
    NSInteger storagePort = [RMRpcSettings storagePort];

    NSURLComponents *components = [[NSURLComponents alloc] init];
    components.scheme = @"http";
    components.host = host;
    components.port = @(self.desiredCoordinatorPort);
    components.path = @"/announce";
    NSMutableArray<NSURLQueryItem *> *items = [NSMutableArray array];
    [items addObject:[NSURLQueryItem queryItemWithName:@"id" value:self.desiredDeviceId ?: @""]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"port" value:[NSString stringWithFormat:@"%ld", (long)servicePort]]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"storage_port" value:[NSString stringWithFormat:@"%ld", (long)storagePort]]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"ip" value:ip]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"model" value:hardwareModel]];
    [items addObject:[NSURLQueryItem queryItemWithName:@"max_size"
                                                 value:[NSString stringWithFormat:@"%llu", (unsigned long long)[LlamaBridge availableProcessMemoryBytes]]]];
    NSString *trimmedNickname = [[self.desiredNickname ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
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
        return 10.0;
    }

    __block NSTimeInterval interval = 10.0;
    dispatch_semaphore_t sema = dispatch_semaphore_create(0);
    NSURLRequest *request = [NSURLRequest requestWithURL:url cachePolicy:NSURLRequestReloadIgnoringLocalCacheData timeoutInterval:5.0];
    [[[NSURLSession sharedSession] dataTaskWithRequest:request completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
        if (error == nil) {
            NSHTTPURLResponse *http = (NSHTTPURLResponse *)response;
            if (http.statusCode == 200) {
                self.lastRuntimeError = @"";
                NSDictionary *json = [NSJSONSerialization JSONObjectWithData:data ?: [NSData data] options:0 error:nil];
                NSNumber *intervalValue = json[@"interval"];
                if (intervalValue != nil) {
                    interval = MAX(intervalValue.doubleValue, 0.5);
                }
                [RMAppLogger logWithTag:@"GENERAL" message:[NSString stringWithFormat:@"announce.ok interval_sec=%.1f", interval]];
            } else {
                self.lastRuntimeError = @"Coordinator announce failed";
                [RMAppLogger logWithLevel:@"WARN" tag:@"GENERAL" message:[NSString stringWithFormat:@"announce.failed status=%ld", (long)http.statusCode]];
            }
        } else {
            self.lastRuntimeError = [NSString stringWithFormat:@"Coordinator announce error: %@", error.localizedDescription];
            [RMAppLogger logWithLevel:@"WARN" tag:@"GENERAL" message:self.lastRuntimeError];
        }
        dispatch_semaphore_signal(sema);
    }] resume];
    dispatch_semaphore_wait(sema, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(6.0 * NSEC_PER_SEC)));
    return interval;
}

- (NSString *)currentRuntimeStatusName {
    switch (self.rpcServerState) {
        case RMRPCServerStateIdle:
            return @"idle";
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
    }
    return @"idle";
}

- (void)publishRuntimeHealthWithStatusOverride:(NSString *)statusOverride {
    NSString *status = statusOverride ?: [self currentRuntimeStatusName];
    NSString *coordinator = self.desiredCoordinatorHost.length > 0
        ? [NSString stringWithFormat:@"%@:%ld", self.desiredCoordinatorHost, (long)self.desiredCoordinatorPort]
        : @"";
    [RMAppLogger rpcHealthWithStatus:status details:@{
        @"endpoint": self.currentRPCEndpoint ?: @"",
        @"coordinator": coordinator,
        @"last_error": self.lastRuntimeError ?: @"",
        @"rpc_available": @([LlamaBridge rpcAvailable]),
        @"discovery_active": @(self.discoveryActive),
        @"rpc_healthy": @(self.rpcHealthy),
        @"storage_healthy": @(self.storageHealthy),
        @"announce_eligible": @(self.announceEligible)
    }];
}

+ (BOOL)probeTCPHost:(NSString *)host port:(NSInteger)port timeout:(NSTimeInterval)timeout {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return NO;
    }

    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host.UTF8String, &addr.sin_addr) != 1) {
        close(sock);
        return NO;
    }

    int result = connect(sock, (struct sockaddr *)&addr, sizeof(addr));
    if (result == 0) {
        close(sock);
        return YES;
    }
    if (errno != EINPROGRESS) {
        close(sock);
        return NO;
    }

    fd_set writeSet;
    FD_ZERO(&writeSet);
    FD_SET(sock, &writeSet);
    struct timeval tv;
    tv.tv_sec = (int)timeout;
    tv.tv_usec = (int)((timeout - (double)(int)timeout) * 1000000.0);
    if (select(sock + 1, NULL, &writeSet, NULL, &tv) <= 0) {
        close(sock);
        return NO;
    }

    int soError = 0;
    socklen_t len = sizeof(soError);
    getsockopt(sock, SOL_SOCKET, SO_ERROR, &soError, &len);
    close(sock);
    return soError == 0;
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
