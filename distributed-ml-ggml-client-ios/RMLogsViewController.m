#import "RMLogsViewController.h"
#import "Diagnostics/AppDiagnostics.h"

typedef NS_ENUM(NSInteger, RMLogCategory) {
    RMLogCategoryRPC = 0,
    RMLogCategoryStorage,
    RMLogCategoryGeneral
};

@interface RMLogsViewController ()

@property (nonatomic, strong) UILabel *healthLabel;
@property (nonatomic, strong) UISegmentedControl *filterControl;
@property (nonatomic, strong) UITextView *logTextView;

@end

@implementation RMLogsViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor whiteColor];
    self.title = @"Logs";

    self.healthLabel = [[UILabel alloc] init];
    self.healthLabel.numberOfLines = 0;
    self.healthLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    self.healthLabel.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.healthLabel];

    self.filterControl = [[UISegmentedControl alloc] initWithItems:@[ @"RPC", @"Storage", @"General" ]];
    self.filterControl.selectedSegmentIndex = 0;
    [self.filterControl addTarget:self action:@selector(refreshLogs) forControlEvents:UIControlEventValueChanged];
    self.filterControl.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.filterControl];

    self.logTextView = [[UITextView alloc] init];
    self.logTextView.editable = NO;
    self.logTextView.backgroundColor = [UIColor blackColor];
    self.logTextView.textColor = [UIColor colorWithRed:0.2 green:0.9 blue:0.3 alpha:1.0];
    self.logTextView.font = [UIFont fontWithName:@"Menlo-Regular" size:12.0] ?: [UIFont systemFontOfSize:12.0];
    self.logTextView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.logTextView];

    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
    NSLayoutYAxisAnchor *topAnchor = self.topLayoutGuide.bottomAnchor;
    NSLayoutYAxisAnchor *bottomAnchor = self.bottomLayoutGuide.topAnchor;
    #pragma clang diagnostic pop
    if (@available(iOS 11.0, *)) {
        topAnchor = self.view.safeAreaLayoutGuide.topAnchor;
        bottomAnchor = self.view.safeAreaLayoutGuide.bottomAnchor;
    }

    [NSLayoutConstraint activateConstraints:@[
        [self.healthLabel.topAnchor constraintEqualToAnchor:topAnchor constant:12.0],
        [self.healthLabel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:16.0],
        [self.healthLabel.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-16.0],

        [self.filterControl.topAnchor constraintEqualToAnchor:self.healthLabel.bottomAnchor constant:12.0],
        [self.filterControl.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:16.0],
        [self.filterControl.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-16.0],

        [self.logTextView.topAnchor constraintEqualToAnchor:self.filterControl.bottomAnchor constant:12.0],
        [self.logTextView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.logTextView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.logTextView.bottomAnchor constraintEqualToAnchor:bottomAnchor],
    ]];

    [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(refreshLogs) name:AppDiagnosticsDidUpdateNotification object:nil];
    [self refreshLogs];
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (RMLogCategory)selectedCategory {
    switch (self.filterControl.selectedSegmentIndex) {
        case 1:
            return RMLogCategoryStorage;
        case 2:
            return RMLogCategoryGeneral;
        default:
            return RMLogCategoryRPC;
    }
}

- (BOOL)line:(NSString *)line matchesCategory:(RMLogCategory)category {
    if ([line containsString:@"[STORAGE]"]) {
        return category == RMLogCategoryStorage;
    }
    if ([line containsString:@"[RPC SERVER]"]) {
        return category == RMLogCategoryRPC;
    }
    return category == RMLogCategoryGeneral;
}

- (void)refreshLogs {
    NSDictionary *health = [AppDiagnostics rpcHealthSnapshot];
    NSString *status = health[@"status"] ?: @"idle";
    NSString *error = health[@"last_error"];
    if ([error isKindOfClass:[NSString class]] && error.length > 0) {
        self.healthLabel.text = [NSString stringWithFormat:@"%@\n%@", [status capitalizedString], error];
    } else {
        self.healthLabel.text = [status capitalizedString];
    }

    RMLogCategory category = [self selectedCategory];
    NSArray<NSString *> *lines = [[AppDiagnostics logsSnapshot] componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSMutableArray<NSString *> *filtered = [NSMutableArray array];
    for (NSString *line in lines) {
        if (line.length == 0) {
            continue;
        }
        if ([self line:line matchesCategory:category]) {
            [filtered addObject:line];
        }
    }
    self.logTextView.text = [filtered componentsJoinedByString:@"\n"];
    if (self.logTextView.text.length > 0) {
        NSRange range = NSMakeRange(self.logTextView.text.length - 1, 1);
        [self.logTextView scrollRangeToVisible:range];
    }
}

@end
