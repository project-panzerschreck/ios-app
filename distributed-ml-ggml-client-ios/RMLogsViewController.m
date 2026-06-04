#import "RMLogsViewController.h"
#import "Diagnostics/AppDiagnostics.h"
#import "RMRpcSettings.h"

typedef NS_ENUM(NSInteger, RMLogCategory) {
    RMLogCategoryRPC = 0,
    RMLogCategoryStorage,
    RMLogCategoryGeneral
};

@interface RMLogsViewController () <UITextViewDelegate>

@property (nonatomic, strong) UILabel *healthLabel;
@property (nonatomic, strong) UILabel *verboseLabel;
@property (nonatomic, strong) UISwitch *verboseSwitch;
@property (nonatomic, strong) UIScrollView *filterScrollView;
@property (nonatomic, strong) UIStackView *filterStack;
@property (nonatomic, strong) UIButton *rpcFilterButton;
@property (nonatomic, strong) UIButton *storageFilterButton;
@property (nonatomic, strong) UIButton *generalFilterButton;
@property (nonatomic, strong) UIView *filterDivider;
@property (nonatomic, strong) UITextView *logTextView;

@property (nonatomic, strong) NSMutableIndexSet *activeFilters;
@property (nonatomic, assign) BOOL autoScrollEnabled;
@property (nonatomic, strong) RMRpcSettings *settings;

@end

@implementation RMLogsViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    self.view.backgroundColor = [UIColor whiteColor];
    self.title = @"Logs";
    self.settings = [RMRpcSettings sharedSettings];
    self.activeFilters = [NSMutableIndexSet indexSet];
    [self.activeFilters addIndex:RMLogCategoryRPC];
    [self.activeFilters addIndex:RMLogCategoryStorage];
    [self.activeFilters addIndex:RMLogCategoryGeneral];
    self.autoScrollEnabled = YES;

    self.healthLabel = [[UILabel alloc] init];
    self.healthLabel.numberOfLines = 0;
    self.healthLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleHeadline];
    self.healthLabel.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.healthLabel];

    self.verboseLabel = [[UILabel alloc] init];
    self.verboseLabel.text = @"Verbose RPC / GGML logs";
    self.verboseLabel.font = [UIFont preferredFontForTextStyle:UIFontTextStyleSubheadline];
    self.verboseLabel.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.verboseLabel];

    self.verboseSwitch = [[UISwitch alloc] init];
    self.verboseSwitch.on = self.settings.isVerboseRPCLoggingEnabled;
    [self.verboseSwitch addTarget:self action:@selector(verboseSwitchChanged:) forControlEvents:UIControlEventValueChanged];
    self.verboseSwitch.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.verboseSwitch];

    self.filterScrollView = [[UIScrollView alloc] init];
    self.filterScrollView.showsHorizontalScrollIndicator = NO;
    self.filterScrollView.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.filterScrollView];

    self.filterStack = [[UIStackView alloc] init];
    self.filterStack.axis = UILayoutConstraintAxisHorizontal;
    self.filterStack.spacing = 10.0;
    self.filterStack.alignment = UIStackViewAlignmentCenter;
    self.filterStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.filterScrollView addSubview:self.filterStack];

    self.rpcFilterButton = [self filterButtonWithTitle:@"RPC" category:RMLogCategoryRPC];
    self.storageFilterButton = [self filterButtonWithTitle:@"Storage" category:RMLogCategoryStorage];
    self.generalFilterButton = [self filterButtonWithTitle:@"General" category:RMLogCategoryGeneral];
    [self.filterStack addArrangedSubview:self.rpcFilterButton];
    [self.filterStack addArrangedSubview:self.storageFilterButton];
    [self.filterStack addArrangedSubview:self.generalFilterButton];

    self.filterDivider = [[UIView alloc] init];
    self.filterDivider.backgroundColor = [UIColor colorWithWhite:0.85 alpha:1.0];
    self.filterDivider.translatesAutoresizingMaskIntoConstraints = NO;
    [self.view addSubview:self.filterDivider];

    self.logTextView = [[UITextView alloc] init];
    self.logTextView.editable = NO;
    self.logTextView.selectable = YES;
    self.logTextView.backgroundColor = [UIColor whiteColor];
    self.logTextView.textColor = [UIColor colorWithRed:0.05 green:0.45 blue:0.18 alpha:1.0];
    self.logTextView.font = [UIFont fontWithName:@"Menlo-Regular" size:12.0] ?: [UIFont systemFontOfSize:12.0];
    self.logTextView.textContainerInset = UIEdgeInsetsMake(12, 12, 12, 12);
    self.logTextView.translatesAutoresizingMaskIntoConstraints = NO;
    self.logTextView.delegate = self;
    [self.view addSubview:self.logTextView];

    NSLayoutYAxisAnchor *topAnchor = self.topLayoutGuide.bottomAnchor;
    NSLayoutYAxisAnchor *bottomAnchor = self.bottomLayoutGuide.topAnchor;

    [NSLayoutConstraint activateConstraints:@[
        [self.healthLabel.topAnchor constraintEqualToAnchor:topAnchor constant:12.0],
        [self.healthLabel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:16.0],
        [self.healthLabel.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-16.0],

        [self.verboseLabel.topAnchor constraintEqualToAnchor:self.healthLabel.bottomAnchor constant:10.0],
        [self.verboseLabel.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor constant:16.0],
        [self.verboseSwitch.centerYAnchor constraintEqualToAnchor:self.verboseLabel.centerYAnchor],
        [self.verboseSwitch.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor constant:-16.0],

        [self.filterScrollView.topAnchor constraintEqualToAnchor:self.verboseLabel.bottomAnchor constant:10.0],
        [self.filterScrollView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.filterScrollView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.filterScrollView.heightAnchor constraintEqualToConstant:44.0],

        [self.filterStack.topAnchor constraintEqualToAnchor:self.filterScrollView.topAnchor constant:6.0],
        [self.filterStack.leadingAnchor constraintEqualToAnchor:self.filterScrollView.leadingAnchor constant:16.0],
        [self.filterStack.trailingAnchor constraintEqualToAnchor:self.filterScrollView.trailingAnchor constant:-16.0],
        [self.filterStack.bottomAnchor constraintEqualToAnchor:self.filterScrollView.bottomAnchor constant:-6.0],
        [self.filterStack.heightAnchor constraintEqualToAnchor:self.filterScrollView.heightAnchor constant:-12.0],

        [self.filterDivider.topAnchor constraintEqualToAnchor:self.filterScrollView.bottomAnchor],
        [self.filterDivider.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.filterDivider.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.filterDivider.heightAnchor constraintEqualToConstant:1.0 / [UIScreen mainScreen].scale],

        [self.logTextView.topAnchor constraintEqualToAnchor:self.filterDivider.bottomAnchor],
        [self.logTextView.leadingAnchor constraintEqualToAnchor:self.view.leadingAnchor],
        [self.logTextView.trailingAnchor constraintEqualToAnchor:self.view.trailingAnchor],
        [self.logTextView.bottomAnchor constraintEqualToAnchor:bottomAnchor],
    ]];

    [[NSNotificationCenter defaultCenter] addObserver:self
                                             selector:@selector(refreshLogs)
                                                 name:AppDiagnosticsDidUpdateNotification
                                               object:nil];
    [self refreshLogs];
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (UIButton *)filterButtonWithTitle:(NSString *)title category:(RMLogCategory)category {
    UIButton *button = [UIButton buttonWithType:UIButtonTypeCustom];
    button.tag = category;
    [button setTitle:title forState:UIControlStateNormal];
    button.titleLabel.font = [UIFont boldSystemFontOfSize:12.0];
    button.contentEdgeInsets = UIEdgeInsetsMake(7, 12, 7, 12);
    button.layer.cornerRadius = 14.0;
    button.layer.masksToBounds = YES;
    button.layer.borderWidth = 1.5;
    [button addTarget:self action:@selector(filterTapped:) forControlEvents:UIControlEventTouchUpInside];
    return button;
}

- (void)filterTapped:(UIButton *)sender {
    RMLogCategory category = (RMLogCategory)sender.tag;
    if ([self.activeFilters containsIndex:category]) {
        if (self.activeFilters.count > 1) {
            [self.activeFilters removeIndex:category];
        }
    } else {
        [self.activeFilters addIndex:category];
    }
    [self updateFilterButtonStyles];
    [self refreshLogs];
}

- (void)updateFilterButtonStyles {
    NSDictionary *health = [AppDiagnostics rpcHealthSnapshot];
    [self styleFilterButton:self.rpcFilterButton
                   category:RMLogCategoryRPC
                     health:health];
    [self styleFilterButton:self.storageFilterButton
                   category:RMLogCategoryStorage
                     health:health];
    [self styleFilterButton:self.generalFilterButton
                   category:RMLogCategoryGeneral
                     health:health];
}

- (void)styleFilterButton:(UIButton *)button
                 category:(RMLogCategory)category
                   health:(NSDictionary *)health {
    BOOL isActive = [self.activeFilters containsIndex:category];
    if (isActive) {
        button.backgroundColor = [UIColor blackColor];
        [button setTitleColor:[UIColor whiteColor] forState:UIControlStateNormal];
    } else {
        button.backgroundColor = [UIColor colorWithWhite:0.9 alpha:1.0];
        [button setTitleColor:[UIColor darkGrayColor] forState:UIControlStateNormal];
    }

    UIColor *outline = [self healthOutlineColorForCategory:category health:health];
    button.layer.borderColor = outline.CGColor;
}

- (UIColor *)healthOutlineColorForCategory:(RMLogCategory)category health:(NSDictionary *)health {
    NSString *status = health[@"status"];
    if (![status isKindOfClass:[NSString class]]) {
        return [UIColor clearColor];
    }
    NSSet<NSString *> *activeStatuses = [NSSet setWithObjects:@"starting", @"running", @"recovering", @"degraded", @"unavailable", nil];
    if (![activeStatuses containsObject:status]) {
        return [UIColor clearColor];
    }

    BOOL isHealthy = YES;
    switch (category) {
        case RMLogCategoryRPC:
            isHealthy = [health[@"rpc_healthy"] boolValue];
            break;
        case RMLogCategoryStorage:
            isHealthy = [health[@"storage_healthy"] boolValue];
            break;
        case RMLogCategoryGeneral:
            isHealthy = [health[@"announce_eligible"] boolValue];
            break;
    }
    return isHealthy ? [UIColor colorWithRed:0.2 green:0.78 blue:0.35 alpha:1.0] : [UIColor colorWithRed:0.9 green:0.25 blue:0.2 alpha:1.0];
}

- (RMLogCategory)categoryForLine:(NSString *)line {
    if ([line containsString:@"[STORAGE]"]) {
        return RMLogCategoryStorage;
    }
    if ([line containsString:@"[RPC SERVER]"] || [line containsString:@"[GGML]"]) {
        return RMLogCategoryRPC;
    }
    return RMLogCategoryGeneral;
}

- (void)verboseSwitchChanged:(UISwitch *)sender {
    self.settings.verboseRPCLogging = sender.isOn;
}

- (NSString *)filteredLogsText {
    NSArray<NSString *> *lines = [[AppDiagnostics logsSnapshot] componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
    NSMutableArray<NSString *> *filtered = [NSMutableArray array];
    for (NSString *line in lines) {
        if (line.length == 0) {
            continue;
        }
        RMLogCategory category = [self categoryForLine:line];
        if ([self.activeFilters containsIndex:category]) {
            [filtered addObject:line];
        }
    }
    return [filtered componentsJoinedByString:@"\n"];
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

    [self updateFilterButtonStyles];

    NSString *text = [self filteredLogsText];
    if (![self.logTextView.text isEqualToString:text]) {
        self.logTextView.text = text;
    }
    if (self.autoScrollEnabled && text.length > 0) {
        NSRange end = NSMakeRange(text.length - 1, 1);
        [self.logTextView scrollRangeToVisible:end];
    }
}

#pragma mark - UIScrollViewDelegate

- (void)scrollViewDidScroll:(UIScrollView *)scrollView {
    if (scrollView != self.logTextView) {
        return;
    }
    CGFloat bottomOffset = scrollView.contentSize.height - (scrollView.contentOffset.y + scrollView.bounds.size.height);
    self.autoScrollEnabled = bottomOffset < 80.0;
}

@end
