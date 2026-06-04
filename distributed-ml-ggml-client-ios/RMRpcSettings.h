#import <Foundation/Foundation.h>

@interface RMRpcSettings : NSObject

@property (nonatomic, copy) NSString *nickname;
@property (nonatomic, assign) NSInteger threads;
@property (nonatomic, copy) NSString *deviceId;
@property (nonatomic, copy) NSString *clusterServerHost;
@property (nonatomic, assign) NSInteger clusterServerPort;
@property (nonatomic, assign, getter=isVerboseRPCLoggingEnabled) BOOL verboseRPCLogging;

+ (instancetype)sharedSettings;
+ (NSInteger)defaultClusterServerPort;
- (void)persistClusterConnection;
- (void)persistVerboseRPCLogging;
+ (NSString *)listenHost;
+ (NSInteger)listenPort;
+ (NSInteger)storagePort;
- (NSURL *)storageDirectory;

@end
