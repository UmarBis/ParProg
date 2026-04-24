#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;

// CUDA ядро для умножения матриц
__global__ void multiplyMatrixGPU(const double* A, const double* B, double* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) {
        double sum = 0.0;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

//чтение
vector<double> readMatrix(string filename, int &size) {
    ifstream file(filename);
    file >> size;
    
    vector<double> matrix(size * size);
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            file >> matrix[i * size + j];
    
    return matrix;
}

// запись
void writeMatrix(string filename, const vector<double>& matrix, int size) {
    ofstream file(filename);
    file << size << endl;
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            file << matrix[i * size + j] << " ";
        file << endl;
    }
}

// CPU версия умножения для сравнения
vector<double> multiplyMatrixCPU(const vector<double>& A, const vector<double>& B, int size) {
    vector<double> C(size * size, 0.0);
    
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                C[i * size + j] += A[i * size + k] * B[k * size + j];
    
    return C;
}

// GPU версия умножения с настраиваемыми параметрами
vector<double> multiplyMatrixGPU(const vector<double>& A, const vector<double>& B, int size, 
                                  int blockSizeX, int blockSizeY, dim3 &gridSize, dim3 &blockSize) {
    vector<double> C(size * size, 0.0);
    
    // Выделение памяти 
    double *d_A, *d_B, *d_C;
    size_t bytes = size * size * sizeof(double);
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Копирование данных
    cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);
    
    // Настройка сетки блоков
    blockSize = dim3(blockSizeX, blockSizeY);
    gridSize = dim3((size + blockSize.x - 1) / blockSize.x, 
                    (size + blockSize.y - 1) / blockSize.y);
    
    // Запуск ядра
    multiplyMatrixGPU<<<gridSize, blockSize>>>(d_A, d_B, d_C, size);
    cudaDeviceSynchronize();
    
    // Копирование результата обратно
    cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Очистка памяти
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}

int main(int argc, char* argv[]) {
    int size1, size2;
    
    // Параметры блоков по умолчанию
    int blockSizeX = 16;
    int blockSizeY = 16;
    
    // Чтение параметров из командной строки
    if (argc >= 3) {
        blockSizeX = atoi(argv[1]);
        blockSizeY = atoi(argv[2]);
    }
    
    auto A = readMatrix("A.txt", size1);
    auto B = readMatrix("B.txt", size2);
    
    if (size1 != size2) {
        cout << "Matrix sizes don't match!\n";
        return 1;
    }
    
    cout << "\n=== Matrix Multiplication ===" << endl;
    cout << "Matrix size: " << size1 << "x" << size1 << endl;
    cout << "Block configuration: " << blockSizeX << "x" << blockSizeY << endl;
    
    // CPU вычисление (опционально для сравнения)
    cout << "\n--- CPU Calculation ---" << endl;
    auto startCPU = chrono::high_resolution_clock::now();
    auto C_cpu = multiplyMatrixCPU(A, B, size1);
    auto endCPU = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedCPU = endCPU - startCPU;
    cout << "CPU time: " << elapsedCPU.count() << " seconds" << endl;
    
    // GPU вычисление
    cout << "\n--- GPU Calculation ---" << endl;
    dim3 gridSize, blockSize;
    auto startGPU = chrono::high_resolution_clock::now();
    auto C_gpu = multiplyMatrixGPU(A, B, size1, blockSizeX, blockSizeY, gridSize, blockSize);
    auto endGPU = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsedGPU = endGPU - startGPU;
    
    cout << "GPU time: " << elapsedGPU.count() << " seconds" << endl;
    cout << "Speedup: " << elapsedCPU.count() / elapsedGPU.count() << "x" << endl;
    cout << "Grid configuration: " << gridSize.x << "x" << gridSize.y << " blocks" << endl;
    cout << "Block configuration: " << blockSize.x << "x" << blockSize.y << " threads" << endl;
    
    // Проверка
    bool correct = true;
    for (int i = 0; i < size1 * size1; i++) {
        if (abs(C_cpu[i] - C_gpu[i]) > 1e-6) {
            correct = false;
            break;
        }
    }
    cout << "\nResult correct: " << (correct ? "Yes" : "No") << endl;
    
    writeMatrix("result_cuda.txt", C_gpu, size1);
    
    long operations = 2LL * size1 * size1 * size1;
    cout << "\nTotal operations: " << operations << endl;
    cout << "GFLOPS: " << (operations / elapsedGPU.count() / 1e9) << endl;
    
    return 0;
}
