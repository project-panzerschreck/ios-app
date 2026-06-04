#import "RMRpcSettings.h"
#import "Bridge/LlamaBridge.h"
#import <UIKit/UIKit.h>

static NSString * const RMNicknameKey = @"rpcNickname";
static NSString * const RMThreadsKey = @"rpcThreads";
static NSString * const RMDeviceIdKey = @"rpcDeviceId";
static NSString * const RMClusterServerHostKey = @"clusterServerHost";
static NSString * const RMClusterServerPortKey = @"clusterServerPort";
static NSString * const RMVerboseRPCLoggingKey = @"verboseRPCLogging";
static NSString * const RMLegacyDiscoveryIpKey = @"rpcDiscoveryIp";
static NSString * const RMLegacyDiscoveryPortKey = @"rpcDiscoveryPort";
static NSString * const RMLegacyDeviceLabelKey = @"clusterDeviceLabel";

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
        NSString *legacyLabel = [defaults stringForKey:RMLegacyDeviceLabelKey];
        _nickname = [[defaults stringForKey:RMNicknameKey] ?: legacyLabel ?: @"" copy];

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

        NSString *legacyHost = [defaults stringForKey:RMLegacyDiscoveryIpKey];
        _clusterServerHost = [[defaults stringForKey:RMClusterServerHostKey] ?: legacyHost ?: @"" copy];
        _clusterServerPort = [defaults integerForKey:RMClusterServerPortKey];
        if (_clusterServerPort == 0) {
            NSInteger legacyPort = [defaults integerForKey:RMLegacyDiscoveryPortKey];
            _clusterServerPort = legacyPort > 0 ? legacyPort : [RMRpcSettings defaultClusterServerPort];
        }

#if defined(VERBOSE_RPC_DEFAULT)
        if ([defaults objectForKey:RMVerboseRPCLoggingKey] == nil) {
            _verboseRPCLogging = YES;
        } else {
            _verboseRPCLogging = [defaults boolForKey:RMVerboseRPCLoggingKey];
        }
#else
        _verboseRPCLogging = [defaults boolForKey:RMVerboseRPCLoggingKey];
#endif
        [LlamaBridge configureRPCLoggingVerbose:_verboseRPCLogging];
    }
    return self;
}

+ (NSInteger)defaultClusterServerPort {
    return 4917;
}

- (void)setNickname:(NSString *)nickname {
    _nickname = [nickname copy] ?: @"";
    [[NSUserDefaults standardUserDefaults] setObject:_nickname forKey:RMNicknameKey];
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

- (void)setVerboseRPCLogging:(BOOL)verboseRPCLogging {
    _verboseRPCLogging = verboseRPCLogging;
    [self persistVerboseRPCLogging];
    [LlamaBridge configureRPCLoggingVerbose:verboseRPCLogging];
}

- (void)persistClusterConnection {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    [defaults setObject:_clusterServerHost forKey:RMClusterServerHostKey];
    [defaults setInteger:_clusterServerPort forKey:RMClusterServerPortKey];
}

- (void)persistVerboseRPCLogging {
    [[NSUserDefaults standardUserDefaults] setBool:_verboseRPCLogging forKey:RMVerboseRPCLoggingKey];
}

+ (NSString *)listenHost {
    return @"0.0.0.0";
}

+ (NSInteger)listenPort {
    return 47651;
}

+ (NSInteger)storagePort {
    return 47672;
}

- (NSURL *)storageDirectory {
    NSURL *documentsDirectory = [[[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory inDomains:NSUserDomainMask] firstObject];
    return [documentsDirectory URLByAppendingPathComponent:@"StorageApp" isDirectory:YES];
}

@end
