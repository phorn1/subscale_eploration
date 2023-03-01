# SUBSCALE algorithm for GPUs

This implementation of the SUBSCALE algorithm was created as part of a master thesis at Hochschule Offenburg. 

## Compilation
The code can either be compiled as a Visual Studio project or with cMake. In both cases the following libraries have to be installed beforehand:   

* [mlpack](https://github.com/mlpack/mlpack)
* [JSON](https://github.com/nlohmann/json)

It is recommended to use [vcpkg](https://github.com/microsoft/vcpkg) for the installation of these libraries to ensure that all additional dependencies are installed as well. If cMake is used to compile the code, the two `CMakeLists.txt` files have to be adjusted to fit the installation directories of the libraries. 

Check the releases if a compiled version of the application is needed.

## Configuration
When the application is started, the path to a config file has to be passed as a command line argument. The config file allows the user to specify various parameters for the application. The file `config.json` is an example for a config file. 

A parameter that currently can't be altered inside of a config file, is the maximum size of files that are read or written from/to system memory. This is set to 50MB, however if the use of larger files is required, the size can be increased inside of the class `CsvDataHandler`.
