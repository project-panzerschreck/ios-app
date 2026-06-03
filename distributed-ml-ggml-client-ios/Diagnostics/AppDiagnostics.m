#import "AppDiagnostics.h"

NSNotificationName const AppDiagnosticsDidUpdateNotification = @"AppDiagnosticsDidUpdateNotification";

static const NSUInteger kAppDiagnosticsMaxLines = 500;

@interface AppDiagnostics ()
@property (nonatomic, strong) dispatch_queue_t queue;
@property (nonatomic, strong) NSMutableArray<NSString *> *lines;
@property (nonatomic, strong) NSMutableDictionary<NSString *, id> *rpcHealth;
@property (nonatomic, strong) NSDateFormatter *formatter;
@end

@implementation AppDiagnostics

+ (instancetype)shared {
    static AppDiagnostics *sharedInstance;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[AppDiagnostics alloc] initPrivate];
    });
    return sharedInstance;
}

- (instancetype)init {
    [NSException raise:@"Singleton" format:@"Use +[AppDiagnostics shared]."];
    return nil;
}

- (instancetype)initPrivate {
    self = [super init];
    if (self) {
        _queue = dispatch_queue_create("AppDiagnostics.queue", DISPATCH_QUEUE_SERIAL);
        _lines = [NSMutableArray array];
        _rpcHealth = [@{
            @"status": @"idle",
            @"last_transition_at": [self.class iso8601StringFromDate:[NSDate date]]
        } mutableCopy];
        _formatter = [[NSDateFormatter alloc] init];
        _formatter.locale = [NSLocale localeWithLocaleIdentifier:@"en_US_POSIX"];
        _formatter.dateFormat = @"yyyy-MM-dd HH:mm:ss.SSS";
    }
    return self;
}

+ (void)logWithLevel:(NSString *)level
                 tag:(NSString *)tag
             message:(NSString *)message {
    [[AppDiagnostics shared] appendLevel:level tag:tag message:message];
}

- (void)appendLevel:(NSString *)level
                tag:(NSString *)tag
            message:(NSString *)message {
    NSDate *now = [NSDate date];
    dispatch_async(self.queue, ^{
        NSArray<NSString *> *parts = [message componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
        for (NSString *part in parts) {
            NSString *line = [self formatLineAt:now level:level tag:tag message:part ?: @""];
            [self.lines addObject:line];
        }
        if (self.lines.count > kAppDiagnosticsMaxLines) {
            NSRange trim = NSMakeRange(0, self.lines.count - kAppDiagnosticsMaxLines);
            [self.lines removeObjectsInRange:trim];
        }
        [self notifyDidUpdate];
    });
}

+ (NSString *)logsSnapshot {
    return [[AppDiagnostics shared] logsSnapshotInternal];
}

- (NSString *)logsSnapshotInternal {
    __block NSString *snapshot = @"";
    dispatch_sync(self.queue, ^{
        snapshot = [self.lines componentsJoinedByString:@"\n"];
    });
    return snapshot;
}

+ (void)setRPCHealthStatus:(NSString *)status
                   details:(NSDictionary<NSString *, id> *)details {
    [[AppDiagnostics shared] setRPCHealthStatusInternal:status details:details];
}

- (void)setRPCHealthStatusInternal:(NSString *)status
                           details:(NSDictionary<NSString *, id> *)details {
    dispatch_async(self.queue, ^{
        [self.rpcHealth removeAllObjects];
        [self.rpcHealth addEntriesFromDictionary:details ?: @{}];
        self.rpcHealth[@"status"] = status ?: @"unknown";
        self.rpcHealth[@"last_transition_at"] = [self.class iso8601StringFromDate:[NSDate date]];
        [self notifyDidUpdate];
    });
}

+ (NSDictionary<NSString *, id> *)rpcHealthSnapshot {
    return [[AppDiagnostics shared] rpcHealthSnapshotInternal];
}

- (NSDictionary<NSString *, id> *)rpcHealthSnapshotInternal {
    __block NSDictionary<NSString *, id> *snapshot = @{};
    dispatch_sync(self.queue, ^{
        snapshot = [self.rpcHealth copy];
    });
    return snapshot;
}

- (NSString *)formatLineAt:(NSDate *)date
                     level:(NSString *)level
                       tag:(NSString *)tag
                   message:(NSString *)message {
    NSString *timestamp = [self.formatter stringFromDate:date];
    NSMutableString *line = [NSMutableString stringWithFormat:@"%@ [%@]", timestamp, level ?: @"INFO"];
    if (tag.length > 0) {
        [line appendFormat:@" [%@]", tag];
    }
    [line appendFormat:@" %@", message ?: @""];
    return line;
}

+ (NSString *)iso8601StringFromDate:(NSDate *)date {
    static NSISO8601DateFormatter *formatter;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        formatter = [[NSISO8601DateFormatter alloc] init];
        formatter.formatOptions = NSISO8601DateFormatWithInternetDateTime;
    });
    return [formatter stringFromDate:date];
}

- (void)notifyDidUpdate {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[NSNotificationCenter defaultCenter] postNotificationName:AppDiagnosticsDidUpdateNotification object:nil];
    });
}

@end
