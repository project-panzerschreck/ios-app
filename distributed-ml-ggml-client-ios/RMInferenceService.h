#import <Foundation/Foundation.h>
#import "Bridge/LlamaBridge.h"

@class RMChatMessage;

typedef NS_ENUM(NSInteger, RMModelState) {
    RMModelStateUnloaded = 0,
    RMModelStateLoading,
    RMModelStateReady,
    RMModelStateGenerating,
    RMModelStateError
};

typedef NS_ENUM(NSInteger, RMRPCServerState) {
    RMRPCServerStateIdle = 0,
    RMRPCServerStateStarting,
    RMRPCServerStateRunning,
    RMRPCServerStateRecovering,
    RMRPCServerStateDegraded,
    RMRPCServerStateUnavailable
};

extern NSString * const RMInferenceServiceDidUpdateNotification;

@interface RMLocalInterface : NSObject

@property (nonatomic, copy) NSString *interfaceId;
@property (nonatomic, copy) NSString *label;
@property (nonatomic, copy) NSString *ip;

- (instancetype)initWithInterfaceId:(NSString *)interfaceId label:(NSString *)label ip:(NSString *)ip;

@end

@interface RMInferenceService : NSObject

@property (nonatomic, assign, readonly) RMModelState modelState;
@property (nonatomic, assign, readonly) RMRPCServerState rpcServerState;
@property (nonatomic, copy, readonly) NSString *modelName;
@property (nonatomic, copy, readonly) NSString *modelErrorMessage;
@property (nonatomic, copy, readonly) NSString *rpcStatusMessage;
@property (nonatomic, copy, readonly) NSString *rpcEndpoint;
@property (nonatomic, assign, readonly) NSInteger modelLayers;
@property (nonatomic, assign, readonly) double tokensPerSecond;
@property (nonatomic, strong, readonly) LlamaModelInfo *modelInfo;
@property (nonatomic, copy, readonly) NSArray<RMChatMessage *> *chatMessages;

+ (instancetype)sharedService;
+ (NSArray<RMLocalInterface *> *)allLocalIPv4Interfaces;

- (void)loadModelFromURL:(NSURL *)url contextLength:(NSInteger)contextLength;
- (void)unloadModel;
- (void)sendMessage:(NSString *)text maxTokens:(NSInteger)maxTokens temperature:(float)temperature;
- (void)cancelGeneration;
- (void)clearChat;
- (void)startRPCServerWithCoordinatorHost:(NSString *)coordinatorHost
                          coordinatorPort:(NSInteger)coordinatorPort
                                 nickname:(NSString *)nickname
                                   threads:(NSInteger)threads
                                  deviceId:(NSString *)deviceId;
- (void)stopRPCServer;
- (void)handleAppDidBecomeActive;
- (void)handleAppWillResignActive;

@end
