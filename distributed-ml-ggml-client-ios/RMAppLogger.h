#import <Foundation/Foundation.h>

@interface RMAppLogger : NSObject

+ (void)logWithLevel:(NSString *)level tag:(NSString *)tag message:(NSString *)message;
+ (void)logWithTag:(NSString *)tag message:(NSString *)message;
+ (void)rpcHealthWithStatus:(NSString *)status details:(NSDictionary<NSString *, id> *)details;

@end
