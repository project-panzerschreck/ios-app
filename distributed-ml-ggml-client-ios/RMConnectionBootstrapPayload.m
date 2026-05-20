#import "RMConnectionBootstrapPayload.h"

@implementation RMConnectionBootstrapPayload

+ (instancetype)payloadWithRawValue:(NSString *)rawValue {
    NSString *input = [[rawValue ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    if (input.length == 0) {
        return nil;
    }

    NSURLComponents *components = [NSURLComponents componentsWithString:input];
    if (components.URL != nil &&
        [[components.scheme lowercaseString] isEqualToString:@"rmcluster"]) {
        NSDictionary *query = [self queryMapFromComponents:components];
        NSString *serverValue = query[@"url"] ?: query[@"host"];
        NSDictionary *parsedHost = [self parseHostAndPort:serverValue];
        if (parsedHost == nil) {
            return nil;
        }

        RMConnectionBootstrapPayload *payload = [[RMConnectionBootstrapPayload alloc] init];
        payload.host = parsedHost[@"host"];
        id embeddedPortValue = parsedHost[@"port"];
        NSNumber *embeddedPort = [embeddedPortValue isKindOfClass:[NSNumber class]] ? embeddedPortValue : nil;
        NSString *explicitPort = query[@"port"];
        payload.port = explicitPort.length > 0 ? @([explicitPort integerValue]) : embeddedPort;
        payload.token = query[@"token"];
        payload.device = query[@"device"] ?: query[@"label"] ?: query[@"name"];
        return payload;
    }

    if (components.host.length > 0) {
        NSDictionary *query = [self queryMapFromComponents:components];
        RMConnectionBootstrapPayload *payload = [[RMConnectionBootstrapPayload alloc] init];
        payload.host = components.host;
        payload.port = components.port ?: (query[@"port"] ? @([query[@"port"] integerValue]) : nil);
        payload.token = query[@"token"];
        payload.device = query[@"device"] ?: query[@"label"] ?: query[@"name"];
        return payload;
    }

    NSDictionary *parsedHost = [self parseHostAndPort:input];
    if (parsedHost == nil) {
        return nil;
    }

    RMConnectionBootstrapPayload *payload = [[RMConnectionBootstrapPayload alloc] init];
        payload.host = parsedHost[@"host"];
        id portValue = parsedHost[@"port"];
        payload.port = [portValue isKindOfClass:[NSNumber class]] ? portValue : nil;
    return payload;
}

+ (NSDictionary *)queryMapFromComponents:(NSURLComponents *)components {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (NSURLQueryItem *item in components.queryItems ?: @[]) {
        NSString *name = [[item.name ?: @"" lowercaseString] copy];
        if (name.length == 0) {
            continue;
        }
        result[name] = item.value ?: @"";
    }
    return result;
}

+ (NSDictionary *)parseHostAndPort:(NSString *)value {
    NSString *trimmed = [[value ?: @"" stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]] copy];
    if (trimmed.length == 0) {
        return nil;
    }

    NSURLComponents *components = [NSURLComponents componentsWithString:trimmed];
    if (components.host.length > 0) {
        return @{
            @"host" : components.host,
            @"port" : components.port ?: [NSNull null]
        };
    }

    NSURLComponents *fallback = [NSURLComponents componentsWithString:[@"http://" stringByAppendingString:trimmed]];
    if (fallback.host.length == 0) {
        return nil;
    }

    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    result[@"host"] = fallback.host;
    if (fallback.port != nil) {
        result[@"port"] = fallback.port;
    }
    return result;
}

@end
