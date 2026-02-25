pub mod common;

pub mod core {
    pub mod compute {
        pub mod ops;
    }
}

pub mod preprocessing {
    pub mod scaler {
        pub mod standard;
        pub mod minmax;
        pub mod robust;
    }
}

pub mod algorithms {
    pub mod linrg;
    pub mod logrg;
}

pub mod metrics {
    pub mod regressor;
    pub mod classifier;
}
