#import "RMAppLogger.h"
#import "AppDiagnostics.h"

@implementation RMAppLogger

+ (void)log:(NSString *)message {
    [self logWithLevel:@"INFO" tag:@"" message:message];
}

+ (void)logWithLevel:(NSString *)level tag:(NSString *)tag message:(NSString *)message {
    [AppDiagnostics logWithLevel:level tag:tag message:message];
    if (tag.length > 0) {
        NSLog(@"[%@] %@", tag, message);
    } else {
        NSLog(@"%@", message);
    }
}

+ (void)rpcHealthWithStatus:(NSString *)status details:(NSDictionary<NSString *, id> *)details {
    [AppDiagnostics setRPCHealthStatus:status details:details];
}

@end
