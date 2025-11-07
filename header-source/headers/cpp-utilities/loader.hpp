#include <vector>
#include <filesystem>
#include <armadillo>

class DatasetOperations {
    public:
        

        void construct_datasets ();
        void load_datasets ();
        void save_dataset (std::filesystem::path save_path);
};

