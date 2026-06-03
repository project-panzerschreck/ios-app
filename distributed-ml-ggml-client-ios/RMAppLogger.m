#import "RMAppLogger.h"
#import "Diagnostics/AppDiagnostics.h"

@implementation RMAppLogger

+ (void)logWithLevel:(NSString *)level tag:(NSString *)tag message:(NSString *)message {
    [AppDiagnostics logWithLevel:level tag:tag message:message];
    NSLog(@"[%@] %@: %@", level, tag, message);
}

+ (void)logWithTag:(NSString *)tag message:(NSString *)message {
    [self logWithLevel:@"INFO" tag:tag message:message];
}

+ (void)rpcHealthWithStatus:(NSString *)status details:(NSDictionary<NSString *, id> *)details {
    [AppDiagnostics setRPCHealthStatus:status details:details];
    NSLog(@"[RPC_HEALTH] status=%@ details=%@", status, details);
}

@end
