#include <filesystem>
#include <armadillo>

class DatasetOperations {
    public:
        void construct_datasets ();
        void load_dataset ();
        void save_dataset (std::filesystem::path save_path);
};

