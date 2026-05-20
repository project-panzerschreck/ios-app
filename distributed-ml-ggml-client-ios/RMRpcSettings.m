#import "RMRpcSettings.h"
#import <UIKit/UIKit.h>

static NSString * const RMRpcHostKey = @"rpcHost";
static NSString * const RMRpcPortKey = @"rpcPort";
static NSString * const RMStoragePortKey = @"rpcStoragePort";
static NSString * const RMDiscoveryIpKey = @"rpcDiscoveryIp";
static NSString * const RMDiscoveryPortKey = @"rpcDiscoveryPort";
static NSString * const RMThreadsKey = @"rpcThreads";
static NSString * const RMDeviceIdKey = @"rpcDeviceId";
static NSString * const RMClusterServerHostKey = @"clusterServerHost";
static NSString * const RMClusterServerPortKey = @"clusterServerPort";
static NSString * const RMClusterDeviceLabelKey = @"clusterDeviceLabel";
static NSString * const RMClusterTokenKey = @"clusterToken";

@implementation RMRpcSettings

+ (instancetype)sharedSettings {
    static RMRpcSettings *settings;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        settings = [[RMRpcSettings alloc] initPrivate];
    });
    return settings;
}

- (instancetype)init {
    [NSException raise:@"Singleton" format:@"Use +sharedSettings"];
    return nil;
}

- (instancetype)initPrivate {
    self = [super init];
    if (self) {
        NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
        _host = [[defaults stringForKey:RMRpcHostKey] ?: @"0.0.0.0" copy];
        _port = [defaults integerForKey:RMRpcPortKey];
        if (_port == 0) {
            _port = 47651;
        }

        _storagePort = [defaults integerForKey:RMStoragePortKey];
        if (_storagePort == 0) {
            _storagePort = 47672;
        }

        _discoveryIp = [[defaults stringForKey:RMDiscoveryIpKey] ?: @"" copy];
        _discoveryPort = [defaults integerForKey:RMDiscoveryPortKey];
        if (_discoveryPort == 0) {
            _discoveryPort = 50055;
        }

        _threads = [defaults integerForKey:RMThreadsKey];
        if (_threads == 0) {
            _threads = 4;
        }

        NSString *existingDeviceId = [defaults stringForKey:RMDeviceIdKey];
        if (existingDeviceId.length == 0) {
            existingDeviceId = [[[UIDevice currentDevice] identifierForVendor] UUIDString] ?: [[NSUUID UUID] UUIDString];
            [defaults setObject:existingDeviceId forKey:RMDeviceIdKey];
        }
        _deviceId = [existingDeviceId copy];

        _clusterServerHost = [[defaults stringForKey:RMClusterServerHostKey] ?: @"" copy];
        _clusterServerPort = [defaults integerForKey:RMClusterServerPortKey];
        if (_clusterServerPort == 0) {
            _clusterServerPort = 4917;
        }
        _clusterDeviceLabel = [[defaults stringForKey:RMClusterDeviceLabelKey] ?: [UIDevice currentDevice].name ?: @"" copy];
        _clusterToken = [[defaults stringForKey:RMClusterTokenKey] ?: @"" copy];
    }
    return self;
}

- (void)setHost:(NSString *)host {
    _host = [host copy] ?: @"";
    [[NSUserDefaults standardUserDefaults] setObject:_host forKey:RMRpcHostKey];
}

- (void)setPort:(NSInteger)port {
    _port = port;
    [[NSUserDefaults standardUserDefaults] setInteger:port forKey:RMRpcPortKey];
}

- (void)setStoragePort:(NSInteger)storagePort {
    _storagePort = storagePort;
    [[NSUserDefaults standardUserDefaults] setInteger:storagePort forKey:RMStoragePortKey];
}

- (void)setDiscoveryIp:(NSString *)discoveryIp {
    _discoveryIp = [discoveryIp copy] ?: @"";
    [[NSUserDefaults standardUserDefaults] setObject:_discoveryIp forKey:RMDiscoveryIpKey];
}

- (void)setDiscoveryPort:(NSInteger)discoveryPort {
    _discoveryPort = discoveryPort;
    [[NSUserDefaults standardUserDefaults] setInteger:discoveryPort forKey:RMDiscoveryPortKey];
}

- (void)setThreads:(NSInteger)threads {
    _threads = threads;
    [[NSUserDefaults standardUserDefaults] setInteger:threads forKey:RMThreadsKey];
}

- (void)setDeviceId:(NSString *)deviceId {
    _deviceId = [deviceId copy] ?: @"";
    [[NSUserDefaults standardUserDefaults] setObject:_deviceId forKey:RMDeviceIdKey];
}

- (void)setClusterServerHost:(NSString *)clusterServerHost {
    _clusterServerHost = [clusterServerHost copy] ?: @"";
    [[NSUserDefaults standardUserDefaults] setObject:_clusterServerHost forKey:RMClusterServerHostKey];
}

- (void)setClusterServerPort:(NSInteger)clusterServerPort {
    _clusterServerPort = clusterServerPort;
    [[NSUserDefaults standardUserDefaults] setInteger:clusterServerPort forKey:RMClusterServerPortKey];
}

- (void)setClusterDeviceLabel:(NSString *)clusterDeviceLabel {
    _clusterDeviceLabel = [clusterDeviceLabel copy] ?: @"";
    [[NSUserDefaults standardUserDefaults] setObject:_clusterDeviceLabel forKey:RMClusterDeviceLabelKey];
}

- (void)setClusterToken:(NSString *)clusterToken {
    _clusterToken = [clusterToken copy] ?: @"";
    [[NSUserDefaults standardUserDefaults] setObject:_clusterToken forKey:RMClusterTokenKey];
}

- (NSURL *)storageDirectory {
    NSURL *documentsDirectory = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    return [documentsDirectory URLByAppendingPathComponent:@"StorageApp" isDirectory:YES];
}

@end
