#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface RMAppLogger : NSObject

+ (void)log:(NSString *)message;
+ (void)logWithLevel:(NSString *)level tag:(NSString *)tag message:(NSString *)message;
+ (void)rpcHealthWithStatus:(NSString *)status details:(NSDictionary<NSString *, id> *)details;

@end

NS_ASSUME_NONNULL_END
