#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

extern NSNotificationName const AppDiagnosticsDidUpdateNotification;

@interface AppDiagnostics : NSObject

+ (instancetype)shared;

+ (void)logWithLevel:(NSString *)level
                 tag:(NSString *)tag
             message:(NSString *)message;

+ (NSString *)logsSnapshot;

+ (void)setRPCHealthStatus:(NSString *)status
                   details:(NSDictionary<NSString *, id> *)details;

+ (NSDictionary<NSString *, id> *)rpcHealthSnapshot;

@end

NS_ASSUME_NONNULL_END
