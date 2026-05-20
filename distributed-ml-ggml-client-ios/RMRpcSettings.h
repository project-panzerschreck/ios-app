#import <Foundation/Foundation.h>

@interface RMRpcSettings : NSObject

@property (nonatomic, copy) NSString *host;
@property (nonatomic, assign) NSInteger port;
@property (nonatomic, assign) NSInteger storagePort;
@property (nonatomic, copy) NSString *discoveryIp;
@property (nonatomic, assign) NSInteger discoveryPort;
@property (nonatomic, assign) NSInteger threads;
@property (nonatomic, copy) NSString *deviceId;
@property (nonatomic, copy) NSString *clusterServerHost;
@property (nonatomic, assign) NSInteger clusterServerPort;
@property (nonatomic, copy) NSString *clusterDeviceLabel;
@property (nonatomic, copy) NSString *clusterToken;

+ (instancetype)sharedSettings;
- (NSURL *)storageDirectory;

@end
