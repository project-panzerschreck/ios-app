#import "RMLogsViewController.h"
#import "Diagnostics/AppDiagnostics.h"

@interface RMLogsViewController ()

@property (nonatomic, strong) UILabel *healthStatusLabel;
@property (nonatomic, strong) UILabel *healthErrorLabel;
@property (nonatomic, strong) UITextView *logTextView;

@end

@implementation RMLogsViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.title = @"Logs";
    self.view.backgroundColor = [UIColor whiteColor];

    self.healthStatusLabel = [[UILabel alloc] init];
    self.healthStatusLabel.font = [UIFont boldSystemFontOfSize:17.0];
    self.healthStatusLabel.translatesAutoresizingMaskIntoConstraints = NO;

    self.healthErrorLabel = [[UILabel alloc] init];
    self.healthErrorLabel.font = [UIFont systemFontOfSize:12.0];
    self.healthErrorLabel.textColor = [UIColor darkGrayColor];
    self.healthErrorLabel.numberOfLines = 0;
    self.healthErrorLabel.translatesAutoresizingMaskIntoConstraints = NO;

    self.logTextView = [[UITextView alloc] init];
    self.logTextView.editable = NO;
    self.logTextView.font = [UIFont fontWithName:@"Menlo-Regular" size:11.0] ?: [UIFont systemFontOfSize:11.0];
    self.logTextView.backgroundColor = [UIColor blackColor];
    self.logTextView.textColor = [UIColor greenColor];
    self.logTextView.translatesAutoresizingMaskIntoConstraints = NO;

    [self.view addSubview:self.healthStatusLabel];
    [self.view addSubview:self.healthErrorLabel];
    [self.view addSubview:self.logTextView];

    [NSLayoutConstraint activateConstraints:@[
        [self.healthStatusLabel.topAnchor constraintEqualToAnchor:self.topLayoutGuide.bottomAnchor constant:12.0],
        [self.healthStatusLabel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:16.0],
        [self.healthStatusLabel.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-16.0],

        [self.healthErrorLabel.topAnchor constraintEqualToAnchor:self.healthStatusLabel.bottomAnchor constant:4.0],
        [self.healthErrorLabel.leadingAnchor constraintEqualToAnchor:self.healthStatusLabel.leadingAnchor],
        [self.healthErrorLabel.trailingAnchor constraintEqualToAnchor:self.healthStatusLabel.trailingAnchor],

        [self.logTextView.topAnchor constraintEqualToAnchor:self.healthErrorLabel.bottomAnchor constant:12.0],
        [self.logTextView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.logTextView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.logTextView.bottomAnchor constraintEqualToAnchor:self.bottomLayoutGuide.topAnchor],
    ]];

    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(diagnosticsDidUpdate:)
                                                 name:AppDiagnosticsDidUpdateNotification
                                               object:nil];
    [self refreshFromDiagnostics];
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (void)diagnosticsDidUpdate:(NSNotification *)notification {
    dispatch_async(dispatch_get_main_queue(), ^{
        [self refreshFromDiagnostics];
    });
}

- (void)refreshFromDiagnostics {
    NSDictionary<NSString *, id> *health = [AppDiagnostics rpcHealthSnapshot];
    NSString *status = health[@"status"];
    if (![status isKindOfClass:[NSString class]]) {
        status = @"idle";
    }
    self.healthStatusLabel.text = [status capitalizedString];

    NSString *lastError = health[@"last_error"];
    if ([lastError isKindOfClass:[NSString class]] && lastError.length > 0) {
        self.healthErrorLabel.text = lastError;
        self.healthErrorLabel.hidden = NO;
    } else {
        self.healthErrorLabel.text = @"";
        self.healthErrorLabel.hidden = YES;
    }

    self.logTextView.text = [AppDiagnostics logsSnapshot];
    if (self.logTextView.text.length > 0) {
        NSRange end = NSMakeRange(self.logTextView.text.length - 1, 1);
        [self.logTextView scrollRangeToVisible:end];
    }
}

@end
