#import "RMStorageServer.h"
#import "RMAppLogger.h"
#import <CommonCrypto/CommonDigest.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <unistd.h>

static const int64_t RMStorageServerMinFreeBytes = 50LL * 1024LL * 1024LL;

@interface RMStorageServerHealthSnapshot : NSObject

@property (nonatomic, strong) NSDate *timestamp;
@property (nonatomic, copy) NSString *status;
@property (nonatomic, copy) NSArray<NSString *> *badChunks;

@end

@implementation RMStorageServerHealthSnapshot
@end

@interface RMStorageServerRequest : NSObject

@property (nonatomic, copy) NSString *method;
@property (nonatomic, copy) NSString *path;
@property (nonatomic, copy) NSArray<NSURLQueryItem *> *queryItems;
@property (nonatomic, copy) NSDictionary<NSString *, NSString *> *headers;
@property (nonatomic, strong) NSData *leftoverBody;

@end

@implementation RMStorageServerRequest
@end

@interface RMStorageServer ()

@property (nonatomic, strong) NSURL *storageDirectory;
@property (nonatomic, assign) int listeningSocket;
@property (nonatomic, strong) dispatch_queue_t serverQueue;
@property (nonatomic, strong) dispatch_source_t acceptSource;
@property (nonatomic, strong) RMStorageServerHealthSnapshot *healthCache;
@property (nonatomic, assign, readwrite, getter=isRunning) BOOL running;
@property (nonatomic, assign) NSInteger activeRequestCount;
@property (nonatomic, strong) NSLock *activeRequestLock;

@end

@implementation RMStorageServer

- (instancetype)initWithStorageDirectory:(NSURL *)storageDirectory {
    self = [super init];
    if (self) {
        _storageDirectory = storageDirectory;
        _listeningSocket = -1;
        _serverQueue = dispatch_queue_create("rmcluster.storage-server", DISPATCH_QUEUE_CONCURRENT);
        _activeRequestLock = [[NSLock alloc] init];
    }
    return self;
}

- (BOOL)isBusy {
    [self.activeRequestLock lock];
    BOOL busy = self.activeRequestCount > 0;
    [self.activeRequestLock unlock];
    return busy;
}

- (BOOL)startOnPort:(NSInteger)port {
    if (self.isRunning) {
        return YES;
    }

    [self stop];

    NSError *directoryError = nil;
    if (![[NSFileManager defaultManager] createDirectoryAtURL:self.storageDirectory
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:&directoryError]) {
        [RMAppLogger logWithLevel:@"ERROR" tag:@"STORAGE" message:[NSString stringWithFormat:@"storage.mkdir.failed path=%@ error=%@", self.storageDirectory.path, directoryError.localizedDescription ?: @"unknown"]];
        return NO;
    }

    int socketFD = socket(AF_INET, SOCK_STREAM, 0);
    if (socketFD < 0) {
        [RMAppLogger logWithLevel:@"ERROR" tag:@"STORAGE" message:[NSString stringWithFormat:@"storage.socket.failed port=%ld errno=%d (%s)", (long)port, errno, strerror(errno)]];
        return NO;
    }

    int reuse = 1;
    setsockopt(socketFD, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_len = sizeof(addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);

    if (bind(socketFD, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        [RMAppLogger logWithLevel:@"ERROR" tag:@"STORAGE" message:[NSString stringWithFormat:@"storage.bind.failed port=%ld errno=%d (%s)", (long)port, errno, strerror(errno)]];
        close(socketFD);
        return NO;
    }

    if (listen(socketFD, 16) < 0) {
        [RMAppLogger logWithLevel:@"ERROR" tag:@"STORAGE" message:[NSString stringWithFormat:@"storage.listen.failed port=%ld errno=%d (%s)", (long)port, errno, strerror(errno)]];
        close(socketFD);
        return NO;
    }

    fcntl(socketFD, F_SETFL, O_NONBLOCK);

    self.listeningSocket = socketFD;
    self.running = YES;

    dispatch_source_t acceptSource = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ,
                                                            (uintptr_t)socketFD,
                                                            0,
                                                            self.serverQueue);
    self.acceptSource = acceptSource;

    __weak typeof(self) weakSelf = self;
    dispatch_source_set_event_handler(acceptSource, ^{
        [weakSelf acceptPendingConnections];
    });
    dispatch_source_set_cancel_handler(acceptSource, ^{
        if (socketFD >= 0) {
            close(socketFD);
        }
    });
    dispatch_resume(acceptSource);
    [RMAppLogger logWithTag:@"STORAGE" message:[NSString stringWithFormat:@"storage.bind.ok port=%ld path=%@", (long)port, self.storageDirectory.path]];
    return YES;
}

- (void)stop {
    if (!self.isRunning && self.listeningSocket < 0) {
        return;
    }

    self.running = NO;
    self.healthCache = nil;

    if (self.acceptSource != nil) {
        dispatch_source_t source = self.acceptSource;
        self.acceptSource = nil;
        dispatch_source_cancel(source);
    }

    if (self.listeningSocket >= 0) {
        close(self.listeningSocket);
        self.listeningSocket = -1;
    }
}

- (void)acceptPendingConnections {
    if (!self.isRunning || self.listeningSocket < 0) {
        return;
    }

    while (YES) {
        int clientFD = accept(self.listeningSocket, NULL, NULL);
        if (clientFD < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                break;
            }
            break;
        }

        dispatch_async(self.serverQueue, ^{
            @autoreleasepool {
                [self handleClient:clientFD];
            }
        });
    }
}

- (void)handleClient:(int)clientFD {
    [self.activeRequestLock lock];
    self.activeRequestCount += 1;
    [self.activeRequestLock unlock];

    NSMutableData *buffer = [NSMutableData data];
    NSData *marker = [@"\r\n\r\n" dataUsingEncoding:NSUTF8StringEncoding];

    while (YES) {
        NSRange markerRange = [buffer rangeOfData:marker options:0 range:NSMakeRange(0, buffer.length)];
        if (markerRange.location != NSNotFound) {
            NSData *headerData = [buffer subdataWithRange:NSMakeRange(0, markerRange.location)];
            NSUInteger bodyOffset = markerRange.location + markerRange.length;
            NSData *leftoverBody = bodyOffset <= buffer.length ? [buffer subdataWithRange:NSMakeRange(bodyOffset, buffer.length - bodyOffset)] : [NSData data];
            RMStorageServerRequest *request = [self parseRequestHeaderData:headerData leftoverBody:leftoverBody];
            if (request == nil) {
                [self sendResponseOnSocket:clientFD status:400 body:[@"Invalid Header" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
                close(clientFD);
                [self.activeRequestLock lock];
                self.activeRequestCount = MAX(0, self.activeRequestCount - 1);
                [self.activeRequestLock unlock];
                return;
            }
            [self routeRequest:request socket:clientFD];
            close(clientFD);
            [self.activeRequestLock lock];
            self.activeRequestCount = MAX(0, self.activeRequestCount - 1);
            [self.activeRequestLock unlock];
            return;
        }

        uint8_t chunk[64 * 1024];
        ssize_t readCount = recv(clientFD, chunk, sizeof(chunk), 0);
        if (readCount <= 0) {
            close(clientFD);
            [self.activeRequestLock lock];
            self.activeRequestCount = MAX(0, self.activeRequestCount - 1);
            [self.activeRequestLock unlock];
            return;
        }
        [buffer appendBytes:chunk length:(NSUInteger)readCount];
    }
}

- (RMStorageServerRequest *)parseRequestHeaderData:(NSData *)headerData leftoverBody:(NSData *)leftoverBody {
    NSString *headerText = [[NSString alloc] initWithData:headerData encoding:NSUTF8StringEncoding];
    if (headerText.length == 0) {
        return nil;
    }

    NSArray<NSString *> *lines = [headerText componentsSeparatedByString:@"\r\n"];
    NSString *requestLine = lines.firstObject;
    NSArray<NSString *> *parts = [requestLine componentsSeparatedByString:@" "];
    if (parts.count < 2) {
        return nil;
    }

    NSString *method = parts[0];
    NSString *target = parts[1];

    NSMutableDictionary<NSString *, NSString *> *headers = [NSMutableDictionary dictionary];
    for (NSUInteger idx = 1; idx < lines.count; idx += 1) {
        NSString *line = lines[idx];
        NSRange colonRange = [line rangeOfString:@":"];
        if (colonRange.location == NSNotFound) {
            continue;
        }
        NSString *key = [[line substringToIndex:colonRange.location] lowercaseString];
        NSString *value = [[line substringFromIndex:colonRange.location + 1] stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        headers[key] = value ?: @"";
    }

    NSURL *baseURL = [NSURL URLWithString:@"http://localhost"];
    NSURL *targetURL = [NSURL URLWithString:target relativeToURL:baseURL];
    NSURLComponents *components = [NSURLComponents componentsWithURL:targetURL resolvingAgainstBaseURL:YES];
    if (components.path.length == 0) {
        return nil;
    }

    RMStorageServerRequest *request = [[RMStorageServerRequest alloc] init];
    request.method = method;
    request.path = components.path;
    request.queryItems = components.queryItems ?: @[];
    request.headers = headers;
    request.leftoverBody = leftoverBody ?: [NSData data];
    return request;
}

- (void)routeRequest:(RMStorageServerRequest *)request socket:(int)clientFD {
    if ([request.path hasPrefix:@"/chunk/"]) {
        NSString *chunkID = [request.path substringFromIndex:[@"/chunk/" length]];
        if (![self isValidSHA256:chunkID]) {
            if ([request.method isEqualToString:@"PUT"]) {
                NSData *body = [self jsonBody:@{ @"error" : @"bad_id" }];
                [self sendResponseOnSocket:clientFD status:400 body:body contentType:@"application/json"];
            } else {
                [self sendResponseOnSocket:clientFD status:400 body:[@"Invalid chunk ID format" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
            }
            return;
        }

        if ([request.method isEqualToString:@"GET"]) {
            [self handleGetChunkWithID:chunkID socket:clientFD];
            return;
        }
        if ([request.method isEqualToString:@"PUT"]) {
            [self handlePutChunkWithID:chunkID request:request socket:clientFD];
            return;
        }
        if ([request.method isEqualToString:@"DELETE"]) {
            [self handleDeleteChunkWithID:chunkID socket:clientFD];
            return;
        }

        [self sendResponseOnSocket:clientFD status:404 body:[@"Not Found" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
        return;
    }

    if ([request.method isEqualToString:@"GET"] && [request.path isEqualToString:@"/chunks/list"]) {
        [self handleListChunksOnSocket:clientFD];
        return;
    }
    if ([request.method isEqualToString:@"GET"] && [request.path isEqualToString:@"/chunks/healthcheck"]) {
        [self handleHealthCheckWithQueryItems:request.queryItems socket:clientFD];
        return;
    }
    if ([request.method isEqualToString:@"GET"] && [request.path isEqualToString:@"/storage_info"]) {
        [self handleStorageInfoOnSocket:clientFD];
        return;
    }

    [self sendResponseOnSocket:clientFD status:404 body:[@"Not Found" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
}

- (void)handleGetChunkWithID:(NSString *)chunkID socket:(int)clientFD {
    NSURL *fileURL = [self.storageDirectory URLByAppendingPathComponent:chunkID];
    if (![[NSFileManager defaultManager] fileExistsAtPath:fileURL.path]) {
        [self sendResponseOnSocket:clientFD
                            status:404
                              body:[self jsonBody:@{ @"error" : @"not_found" }]
                       contentType:@"application/json"];
        return;
    }

    NSString *actualHash = [self sha256HexForFileURL:fileURL];
    if (actualHash.length == 0 || [actualHash caseInsensitiveCompare:chunkID] != NSOrderedSame) {
        [self sendResponseOnSocket:clientFD
                            status:404
                              body:[self jsonBody:@{ @"error" : @"corrupted_chunk" }]
                       contentType:@"application/json"];
        return;
    }

    NSData *data = [NSData dataWithContentsOfURL:fileURL];
    if (data == nil) {
        [self sendResponseOnSocket:clientFD status:500 body:[@"Read failed" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
        return;
    }

    [self sendResponseOnSocket:clientFD status:200 body:data contentType:@"application/octet-stream"];
}

- (void)handlePutChunkWithID:(NSString *)chunkID request:(RMStorageServerRequest *)request socket:(int)clientFD {
    int64_t contentLength = (int64_t)[request.headers[@"content-length"] longLongValue];
    int64_t available = [self availableBytes];
    if (available - contentLength < RMStorageServerMinFreeBytes) {
        [self sendResponseOnSocket:clientFD
                            status:507
                              body:[self jsonBody:@{ @"error" : @"insufficient_storage" }]
                       contentType:@"application/json"];
        return;
    }

    NSURL *tempURL = [self.storageDirectory URLByAppendingPathComponent:[NSString stringWithFormat:@"%@.tmp", [[NSUUID UUID] UUIDString]]];
    [[NSFileManager defaultManager] createFileAtPath:tempURL.path contents:nil attributes:nil];
    NSFileHandle *handle = [NSFileHandle fileHandleForWritingAtPath:tempURL.path];
    if (handle == nil) {
        [self sendResponseOnSocket:clientFD status:500 body:[@"Write failed" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
        return;
    }

    BOOL success = YES;
    int64_t bytesWritten = 0;
    @try {
        if (request.leftoverBody.length > 0) {
            NSUInteger initialCount = (NSUInteger)MIN((int64_t)request.leftoverBody.length, contentLength);
            NSData *initialChunk = [request.leftoverBody subdataWithRange:NSMakeRange(0, initialCount)];
            [handle writeData:initialChunk];
            bytesWritten += (int64_t)initialChunk.length;
        }

        while (bytesWritten < contentLength) {
            size_t toRead = (size_t)MIN(contentLength - bytesWritten, (int64_t)(1024 * 1024));
            NSMutableData *bodyChunk = [NSMutableData dataWithLength:toRead];
            ssize_t received = recv(clientFD, bodyChunk.mutableBytes, toRead, 0);
            if (received <= 0) {
                success = NO;
                break;
            }
            bodyChunk.length = (NSUInteger)received;
            [handle writeData:bodyChunk];
            bytesWritten += received;
        }
    } @catch (__unused NSException *exception) {
        success = NO;
    }
    [handle closeFile];

    if (!success || bytesWritten != contentLength) {
        [[NSFileManager defaultManager] removeItemAtURL:tempURL error:nil];
        [self sendResponseOnSocket:clientFD status:500 body:[@"Write failed" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
        return;
    }

    NSString *actualHash = [self sha256HexForFileURL:tempURL];
    if (actualHash.length == 0 || [actualHash caseInsensitiveCompare:chunkID] != NSOrderedSame) {
        [[NSFileManager defaultManager] removeItemAtURL:tempURL error:nil];
        [self sendResponseOnSocket:clientFD
                            status:400
                              body:[self jsonBody:@{ @"error" : @"checksum_incorrect" }]
                       contentType:@"application/json"];
        return;
    }

    NSURL *targetURL = [self.storageDirectory URLByAppendingPathComponent:chunkID];
    if ([[NSFileManager defaultManager] fileExistsAtPath:targetURL.path]) {
        [[NSFileManager defaultManager] removeItemAtURL:targetURL error:nil];
    }

    NSError *moveError = nil;
    if (![[NSFileManager defaultManager] moveItemAtURL:tempURL toURL:targetURL error:&moveError]) {
        [[NSFileManager defaultManager] removeItemAtURL:tempURL error:nil];
        [self sendResponseOnSocket:clientFD status:500 body:[@"Write failed" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
        return;
    }

    self.healthCache = nil;
    [self sendResponseOnSocket:clientFD status:200 body:[@"OK" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
}

- (void)handleDeleteChunkWithID:(NSString *)chunkID socket:(int)clientFD {
    NSURL *fileURL = [self.storageDirectory URLByAppendingPathComponent:chunkID];
    if (![[NSFileManager defaultManager] fileExistsAtPath:fileURL.path]) {
        [self sendResponseOnSocket:clientFD
                            status:404
                              body:[self jsonBody:@{ @"error" : @"not_found" }]
                       contentType:@"application/json"];
        return;
    }

    NSError *removeError = nil;
    if (![[NSFileManager defaultManager] removeItemAtURL:fileURL error:&removeError]) {
        [self sendResponseOnSocket:clientFD status:500 body:[@"Delete failed" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
        return;
    }

    self.healthCache = nil;
    [self sendResponseOnSocket:clientFD status:200 body:[@"OK" dataUsingEncoding:NSUTF8StringEncoding] contentType:@"text/plain"];
}

- (void)handleListChunksOnSocket:(int)clientFD {
    NSArray<NSURL *> *files = [[NSFileManager defaultManager] contentsOfDirectoryAtURL:self.storageDirectory
                                                             includingPropertiesForKeys:nil
                                                                                options:0
                                                                                  error:nil] ?: @[];
    NSMutableArray<NSString *> *chunkIDs = [NSMutableArray array];
    for (NSURL *fileURL in files) {
        NSString *name = fileURL.lastPathComponent;
        if ([self isValidSHA256:name]) {
            [chunkIDs addObject:name];
        }
    }

    [self sendResponseOnSocket:clientFD status:200 body:[self jsonBody:chunkIDs] contentType:@"application/json"];
}

- (void)handleHealthCheckWithQueryItems:(NSArray<NSURLQueryItem *> *)queryItems socket:(int)clientFD {
    double maxAgeSeconds = 300.0;
    for (NSURLQueryItem *item in queryItems) {
        if ([item.name isEqualToString:@"max_age"]) {
            maxAgeSeconds = item.value.doubleValue > 0 ? item.value.doubleValue : 300.0;
            break;
        }
    }

    if (self.healthCache != nil && [[NSDate date] timeIntervalSinceDate:self.healthCache.timestamp] < maxAgeSeconds) {
        NSDictionary *cached = @{
            @"status" : self.healthCache.status ?: @"healthy",
            @"bad_chunks" : self.healthCache.badChunks ?: @[]
        };
        [self sendResponseOnSocket:clientFD status:200 body:[self jsonBody:cached] contentType:@"application/json"];
        return;
    }

    NSArray<NSURL *> *files = [[NSFileManager defaultManager] contentsOfDirectoryAtURL:self.storageDirectory
                                                             includingPropertiesForKeys:nil
                                                                                options:0
                                                                                  error:nil] ?: @[];
    NSMutableArray<NSString *> *badChunks = [NSMutableArray array];
    for (NSURL *fileURL in files) {
        NSString *name = fileURL.lastPathComponent;
        if (![self isValidSHA256:name]) {
            continue;
        }
        NSString *actualHash = [self sha256HexForFileURL:fileURL];
        if (actualHash.length == 0 || [actualHash caseInsensitiveCompare:name] != NSOrderedSame) {
            [badChunks addObject:name];
        }
    }

    RMStorageServerHealthSnapshot *snapshot = [[RMStorageServerHealthSnapshot alloc] init];
    snapshot.timestamp = [NSDate date];
    snapshot.status = badChunks.count == 0 ? @"healthy" : @"degraded";
    snapshot.badChunks = badChunks;
    self.healthCache = snapshot;

    NSDictionary *body = @{
        @"status" : snapshot.status,
        @"bad_chunks" : snapshot.badChunks
    };
    [self sendResponseOnSocket:clientFD status:200 body:[self jsonBody:body] contentType:@"application/json"];
}

- (void)handleStorageInfoOnSocket:(int)clientFD {
    NSDictionary *body = @{
        @"total_space" : @([self totalBytes]),
        @"used_space" : @([self usedBytes]),
        @"available_space" : @([self availableBytes])
    };
    [self sendResponseOnSocket:clientFD status:200 body:[self jsonBody:body] contentType:@"application/json"];
}

- (BOOL)isValidSHA256:(NSString *)value {
    if (value.length != 64) {
        return NO;
    }

    NSCharacterSet *hexSet = [NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdefABCDEF"];
    for (NSUInteger idx = 0; idx < value.length; idx += 1) {
        unichar character = [value characterAtIndex:idx];
        if (![hexSet characterIsMember:character]) {
            return NO;
        }
    }
    return YES;
}

- (NSString *)sha256HexForFileURL:(NSURL *)fileURL {
    NSFileHandle *handle = [NSFileHandle fileHandleForReadingAtPath:fileURL.path];
    if (handle == nil) {
        return @"";
    }

    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);

    @try {
        while (YES) {
            NSData *data = [handle readDataOfLength:(NSUInteger)(64 * 1024)];
            if (data.length == 0) {
                break;
            }
            CC_SHA256_Update(&context, data.bytes, (CC_LONG)data.length);
        }
    } @catch (__unused NSException *exception) {
        [handle closeFile];
        return @"";
    }
    [handle closeFile];

    unsigned char digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256_Final(digest, &context);

    NSMutableString *output = [NSMutableString stringWithCapacity:(NSUInteger)(CC_SHA256_DIGEST_LENGTH * 2)];
    for (NSUInteger idx = 0; idx < CC_SHA256_DIGEST_LENGTH; idx += 1) {
        [output appendFormat:@"%02x", digest[idx]];
    }
    return output;
}

- (int64_t)totalBytes {
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfFileSystemForPath:self.storageDirectory.path error:nil];
    return [attributes[NSFileSystemSize] longLongValue];
}

- (int64_t)availableBytes {
    NSDictionary *attributes = [[NSFileManager defaultManager] attributesOfFileSystemForPath:self.storageDirectory.path error:nil];
    return [attributes[NSFileSystemFreeSize] longLongValue];
}

- (int64_t)usedBytes {
    NSArray<NSURL *> *files = [[NSFileManager defaultManager] contentsOfDirectoryAtURL:self.storageDirectory
                                                             includingPropertiesForKeys:@[ NSURLFileSizeKey ]
                                                                                options:0
                                                                                  error:nil] ?: @[];
    int64_t total = 0;
    for (NSURL *fileURL in files) {
        NSNumber *fileSize = nil;
        [fileURL getResourceValue:&fileSize forKey:NSURLFileSizeKey error:nil];
        total += fileSize.longLongValue;
    }
    return total;
}

- (NSData *)jsonBody:(id)value {
    NSData *data = [NSJSONSerialization dataWithJSONObject:value options:0 error:nil];
    return data ?: [@"{}" dataUsingEncoding:NSUTF8StringEncoding];
}

- (void)sendResponseOnSocket:(int)clientFD status:(NSInteger)status body:(NSData *)body contentType:(NSString *)contentType {
    NSString *statusText = [self statusMessageForCode:status];
    NSMutableString *header = [NSMutableString stringWithFormat:@"HTTP/1.1 %ld %@\r\n", (long)status, statusText];
    [header appendFormat:@"Content-Length: %lu\r\n", (unsigned long)body.length];
    [header appendFormat:@"Content-Type: %@\r\n", contentType ?: @"application/octet-stream"];
    [header appendString:@"Connection: close\r\n\r\n"];

    NSMutableData *response = [NSMutableData dataWithData:[header dataUsingEncoding:NSUTF8StringEncoding]];
    [response appendData:body ?: [NSData data]];

    const uint8_t *bytes = response.bytes;
    NSUInteger totalSent = 0;
    while (totalSent < response.length) {
        ssize_t sent = send(clientFD, bytes + totalSent, response.length - totalSent, 0);
        if (sent <= 0) {
            break;
        }
        totalSent += (NSUInteger)sent;
    }
}

- (NSString *)statusMessageForCode:(NSInteger)status {
    switch (status) {
        case 200: return @"OK";
        case 400: return @"Bad Request";
        case 404: return @"Not Found";
        case 500: return @"Internal Server Error";
        case 507: return @"Insufficient Storage";
        default:  return @"Error";
    }
}

@end
