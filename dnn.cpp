#include <iostream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <numeric>
#include <string>
#include <cassert>
#include <vector>
#include <random>

/*

     -----
     |   |
     |   |           -----
     |   |           |   |           -----
     |   |           |   |           |   |
     |784| == W1 ==> |30 | == W2 ==> |10 |
     |   |           |   |           |   |
     |   |           |   |           -----
     |   |           ----- 
     -----  
  input layer     hidden layer     output layer
      (I)             (H)             (O)

  (bias and sigmoid activation ignored)

  data:
    I : N x 784
    W1: 784 x 30
    W2: 30 x 10
    O : N x 10

  W1 and W2 are trainable parameters

  forward:
    I * W1 => H (N x 30)
    H * W2 => O (N x 10)

  backward: 

*/ 

using std::vector;

typedef vector<float> vec_t;


namespace blas {

// Y[i] += ALPHA * X[i]
template <typename T>
void axpy(size_t N, T ALPHA, const T *X, T *Y)
{
    for(size_t i = 0; i < N; ++i) Y[i] += ALPHA*X[i];
}

// vector * vector
template <typename T>
T dot(size_t N, const T *X, const T *Y) {

  T sum = 0;
  for (size_t i = 0; i < N; ++i) {
    sum += X[i] * Y[i];
  }
  return sum;
}

}


template <typename Iter>
void uniform_rand(Iter begin, Iter end, float min, float max) {
  std::mt19937 gen;
  std::uniform_real_distribution<float> dst(min, max);
  for (Iter it = begin; it != end; ++it) *it = dst(gen);
}

inline void zeros(vector<vec_t> &tensor) {
  for (auto &t : tensor) {
    std::fill(t.begin(), t.end(), 0);
  }
}

inline void zeros(vec_t &vec) {
  std::fill(vec.begin(), vec.end(), 0);
}

class layer {
public:
  layer(size_t in_dim,
        size_t out_dim)
      : weights(in_dim*out_dim)
      , weights_grad(in_dim*out_dim)
      , bias(out_dim, 0)
      , bias_grad(out_dim)
      , outputs({vec_t(out_dim)})
      , outputs_grad({vec_t(out_dim)}) {

      const float weight_base = std::sqrt(6.0f / (in_dim + out_dim));
      uniform_rand(weights.begin(), weights.end(), -weight_base, weight_base);
  }

  void save(std::ostream &os) const {

    for (auto w : weights) os << w << " ";
    for (auto w : bias) os << w << " ";
  }

  void load(std::istream &is) {
    for (auto &w : weights) is >> w;
    for (auto &w : bias) is >> w;
  }

  const vector<vec_t>& get_output() const {
    return outputs;
  }

  vector<vec_t>& get_output_grad() {
    return outputs_grad;
  }

  template<typename ITER>
  const vector<vec_t>& forward(ITER inputs, size_t batch_size) {

      outputs.resize(batch_size, outputs[0]);
      outputs_grad.resize(batch_size, outputs_grad[0]);

      zeros(outputs);

      const auto in_dim = inputs->size();
      const auto out_dim = outputs[0].size();

      for(size_t n = 0; n < batch_size; n++, inputs++) {
        auto &in = *inputs;
        auto &out = outputs[n];

        for (size_t i = 0; i < out_dim; i++) {

            for (size_t c = 0; c < in_dim; c++) {
              out[i] += weights[c * out_dim + i] * in[c];
            }
        }

        blas::axpy(out_dim, 1.0f, &bias[0], &out[0]);

        for (size_t i = 0; i < out_dim; i++) {
            // sigmoid
            out[i] = float(1) / (float(1) + std::exp(-out[i]));
        }
      }

      return outputs;
  }

  template<typename ITER>
  void backward(ITER inputs, 
                vector<vec_t>* in_grad) {

    if (in_grad)
      zeros(*in_grad);

    zeros(bias_grad);
    zeros(weights_grad);

    const auto batch_size = outputs.size();
    const auto in_dim = inputs->size();
    const auto out_dim = outputs[0].size();

    for (size_t n = 0; n < batch_size; n++, inputs++) {

        auto &in = *inputs;

        for (size_t j = 0; j < out_dim; j++) {
          // dx = dy * (gradient of sigmoid)
          outputs_grad[n][j] = outputs_grad[n][j] * outputs[n][j] * (float(1) - outputs[n][j]);
        }
      
        if (in_grad) {
          for (size_t c = 0; c < in_dim; c++) {
            // propagate delta to previous layer
            // in_grad[c] += outputs_grad[r] * W[c * out_dim + r]
            (*in_grad)[n][c] += blas::dot(
              out_dim, &outputs_grad[n][0], &weights[c * out_dim]);
          }
        }

        // accumulate weight-step using delta
        // dW[c * out_size + i] += outputs_grad[i] * inputs[c]
        for (size_t c = 0; c < in_dim; c++) {
          blas::axpy(out_dim, in[c], 
                            &outputs_grad[n][0], 
                            &weights_grad[c * out_dim]);
        }

        // bias_grad += outputs_grad[n]
        blas::axpy(out_dim, 1.0f, &outputs_grad[n][0], &bias_grad[0]);
    }

    float rcp_batch_size = float(1.0) / float(batch_size);

    for (auto& g : weights_grad)
      g *= rcp_batch_size;

    for (auto& g : bias_grad)
      g *= rcp_batch_size; 
  }

  void update_weight(float learning_rate, float lambda) {

    for(size_t i=0; i < weights.size(); i++) { 
        weights[i] = weights[i] - learning_rate * (weights_grad[i] + lambda * weights[i]); 
    }

    for(size_t i=0; i < bias.size(); i++) { 
        bias[i] = bias[i] - learning_rate * bias_grad[i]; 
    }
  }

protected:
  vec_t weights;
  vec_t weights_grad;
  vec_t bias;
  vec_t bias_grad;
  vector<vec_t> outputs;
  vector<vec_t> outputs_grad;
};


// mean-squared-error loss function for regression
class mse {
 public:
  static float f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float d{0.0};

    for (size_t i = 0; i < y.size(); ++i) d += (y[i] - t[i]) * (y[i] - t[i]);

    return d / static_cast<float>(y.size());
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());
    float factor = float(2) / static_cast<float>(t.size());

    for (size_t i = 0; i < y.size(); ++i) d[i] = factor * (y[i] - t[i]);

    return d;
  }
};


// cross-entropy loss function for (multiple independent) binary classifications
class cross_entropy {
 public:
  static float f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float d{0};

    for (size_t i = 0; i < y.size(); ++i)
      d += -t[i] * std::log(y[i]) -
           (float(1) - t[i]) * std::log(float(1) - y[i]);

    return d;
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());

    for (size_t i = 0; i < y.size(); ++i)
      d[i]        = (y[i] - t[i]) / (y[i] * (float(1) - y[i]));

    return d;
  }
};


class network {
public:
  network(const vector<int>& dims) {
    for (size_t i=0; i<dims.size()-1; i++) {
      layers_.push_back(layer(dims[i], dims[i+1]));
    }
  }

  template<typename ITER>
  vector<vec_t> forward(ITER inputs, size_t batch_size) {

    for (auto& l : layers_) {
      auto& outputs = l.forward(inputs, batch_size);
      inputs = &outputs[0];
    }
    return layers_.back().get_output();
  }

  template <typename LOSS, typename ITER, typename ITER2>
  void bprop(ITER in_data,
              const vector<vec_t> &out,
              ITER2 truth) {

    const auto batch_size = out.size();

    auto& final_grad = layers_.back().get_output_grad();
    
    for (size_t n = 0; n < batch_size; ++n, truth++) {

      final_grad[n] = LOSS::df(out[n], *truth);
    }

    for (auto l = layers_.rbegin(); l != layers_.rend(); l++) {

      ITER inputs;
      vector<vec_t>* in_grad;

      if ((l+1) == layers_.rend()) {
        inputs = in_data;
        in_grad = nullptr;
      } else {
        inputs = &(l+1)->get_output()[0];
        in_grad = &(l+1)->get_output_grad();
      }
      l->backward(inputs, in_grad);
    }
  }

  vec_t predict(const vec_t &in) { 

    return forward(&in, 1)[0];
  }

  int predict_label(const vec_t &in) { 

    auto out = predict(in);
    return std::max_element(out.begin(), out.end()) - out.begin();
  }

  template <typename LOSS,
            typename OnBatchEnumerate,
            typename OnEpochEnumerate>
  void train(
             const vector<vec_t> &inputs,
             const vector<vec_t> &labels,
             size_t batch_size,
             int epoch,
             float learning_rate,
             float lambda,
             OnBatchEnumerate on_batch_enumerate,
             OnEpochEnumerate on_epoch_enumerate) {

    for (int iter = 0; iter < epoch; iter++) {
      for (size_t i = 0; i < inputs.size(); i += batch_size) {

          auto size = std::min(batch_size, inputs.size() - i);
          bprop<LOSS>(&inputs[i], forward(&inputs[i], size), &labels[i]);
          for (auto& l : layers_) {
            l.update_weight(learning_rate, lambda);
          }

          on_batch_enumerate();
      }
      on_epoch_enumerate();
    }
  }


  void load(const std::string &filename) {

    std::ifstream ifs(filename);
    if (ifs.fail() || ifs.bad()) throw std::runtime_error("failed to open:" + filename);
    ifs.precision(std::numeric_limits<float>::digits10);
    for (auto &l : layers_) {
      l.load(ifs);
    }
  }

  void save(const std::string &filename) const {

    std::ofstream ofs(filename);
    if (ofs.fail() || ofs.bad()) throw std::runtime_error("failed to open:" + filename);

    ofs.precision(std::numeric_limits<float>::digits10);
    for (auto &l : layers_) {
      l.save(ofs);
    }
  }

protected:
  vector<layer> layers_;
};


template<typename LABLE_ARRAY>
static void onehot(const LABLE_ARRAY &labels, vector<vec_t> &vec, const int output_dim) {

    vec.reserve(labels.size());
    for (auto t : labels) {
      assert(t < output_dim);
      vec.emplace_back(output_dim, 0);
      vec.back()[t] = 1;
    }
}

class timer {
 public:
  timer() : t1(std::chrono::high_resolution_clock::now()) {}
  float elapsed() {
    return std::chrono::duration_cast<std::chrono::duration<float>>(
             std::chrono::high_resolution_clock::now() - t1)
      .count();
  }
  void restart() { t1 = std::chrono::high_resolution_clock::now(); }
  void start() { t1 = std::chrono::high_resolution_clock::now(); }
  void stop() { t2 = std::chrono::high_resolution_clock::now(); }
  float total() {
    stop();
    return std::chrono::duration_cast<std::chrono::duration<float>>(t2 - t1)
      .count();
  }
  ~timer() {}

 private:
  std::chrono::high_resolution_clock::time_point t1, t2;
};

template <typename T>
T *reverse_endian(T *p) {
  std::reverse(reinterpret_cast<char *>(p),
               reinterpret_cast<char *>(p) + sizeof(T));
  return p;
}

inline void parse_mnist_labels(const std::string &label_file,
                               vector<int>& labels) {
  std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);

  if (ifs.bad() || ifs.fail())
    throw std::runtime_error("failed to open file:" + label_file);

  uint32_t magic_number, num_items;

  ifs.read(reinterpret_cast<char *>(&magic_number), 4);
  ifs.read(reinterpret_cast<char *>(&num_items), 4);

  reverse_endian(&num_items);

  if (magic_number != 0x01080000 || num_items <= 0)
    throw std::runtime_error("MNIST label-file format error");

  labels.resize(num_items);
  for (uint32_t i = 0; i < num_items; i++) {
    uint8_t label;
    ifs.read(reinterpret_cast<char *>(&label), 1);
    labels[i] = static_cast<int>(label);
  }
}

static void parse_mnist_images(const std::string &image_file,
                               vector<vec_t> &images) {

    std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail())
      throw std::runtime_error("failed to open file:" + image_file);

    uint32_t magic_number;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;

    ifs.read(reinterpret_cast<char *>(&magic_number), 4);
    ifs.read(reinterpret_cast<char *>(&num_items), 4);
    ifs.read(reinterpret_cast<char *>(&num_rows), 4);
    ifs.read(reinterpret_cast<char *>(&num_cols), 4);

    reverse_endian(&num_items);
    reverse_endian(&num_rows);
    reverse_endian(&num_cols);

    if (magic_number != 0x03080000 || num_items <= 0)
      throw std::runtime_error("MNIST label-file format error");

    size_t image_size = num_cols * num_rows;
    vector<uint8_t> image_vec(image_size);

    images.resize(num_items);
    for (uint32_t i = 0; i < num_items; i++) {

      vec_t image(image_size);
      ifs.read(reinterpret_cast<char *>(&image_vec[0]), image_size);

      for (auto idx = size_t{0}; idx < image_size; idx++)
        image[idx] = (image_vec[idx] / 255.0f);
        
      images[i] = image;
    }
}


static void train_mnist(const std::string &data_dir_path,
                        double learning_rate,
                        const int n_train_epochs,
                        const int batch_size) {

  float lambda = 0.00001;

  network nn({784, 30, 10});

  std::cout << "load models..." << std::endl;

  // load MNIST dataset
  vector<int> train_labels, test_labels;
  vector<vec_t> train_images, test_images, truth_labels;

  parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                               train_labels);
  parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                               train_images);
  parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                               test_labels);
  parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                               test_images);

  std::cout << "start training" << std::endl;

  const auto samples_count = train_images.size();
  size_t trained_count = 0;

  timer t;

  int epoch = 1;

  // create callback
  auto on_enumerate_epoch = [&]() {

    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;
    ++epoch;
    trained_count = 0;

    // verify 
    size_t num_correct = 0;
    for (size_t i = 0; i < test_images.size(); i++) {

      if (nn.predict_label(test_images[i]) == test_labels[i]) num_correct++;
    }
    std::cout << "Accuracy: " << num_correct << "/" << test_images.size() << std::endl;

    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { 

    trained_count += batch_size;

    std::ostringstream ss;
    ss << trained_count << "/" << samples_count;

    std::cout.flags(std::ios::left);
    std::cout << std::setw(20) << ss.str();
    for (size_t i=0; i < 20; ++i) std::cout << '\b';
  };

  // training
  onehot(train_labels, truth_labels, 10);
  nn.train<cross_entropy>(train_images, truth_labels, batch_size,
                          n_train_epochs, learning_rate, lambda, 
                          on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  nn.save("weights.txt");
}

static void train_arth(double learning_rate,
                        const int n_train_epochs,
                        char op,
                        int test_x,
                        int test_y) {

  vector<int> train_labels;
  vector<vec_t> train_images, truth_labels;
                         
  int in_max = std::max(test_x, test_y);
  in_max = std::max(in_max, 2);

  constexpr int invalid = 99999999;
  int max_t = 0;
  int min_t = 0;
  for (int x=0; x<=in_max; x++) {
    for (int y=0; y<=in_max; y++) {
      int t;
      switch(op) {
          case '*':
          case 'x':
          case 'X':
            t = x*y;
            op = 'x';
            break;
          case '-':
            t = x-y;
            break;
          case '/':
            t = y==0 ? invalid : x/y;
            break;
          case '+':
          default:
            t = x+y;
            op = '+';
            break;
        }

        train_images.push_back({float(x), float(y)});
        train_labels.push_back(t);

        if (t != invalid) {
          if (t > max_t) {
            max_t = t;
          }
          if (t < min_t) {
            min_t = t;
          }
        }
    }
  }

  const auto range = (max_t-min_t)+2; // one for invalid

  for (auto& t : train_labels) {
    if (t != invalid)
        t -= min_t;
    else 
        t = range - 1;
  }

  network nn({2, 30, range});

  int epoch = 1;

  auto on_enumerate_epoch = [&]() {
    ++epoch;
    std::ostringstream ss;
    ss << epoch << "/" << n_train_epochs;
    std::cout.flags(std::ios::left);
    std::cout << std::setw(20) << ss.str();
    for (size_t i=0; i < 20; ++i) std::cout << '\b';
  };

  auto batch_size = train_images.size();
  onehot(train_labels, truth_labels, range);
  nn.train<cross_entropy>(train_images, truth_labels, batch_size,
                          n_train_epochs, learning_rate, 0.0005f, 
                          []{},
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  auto resutl = nn.predict_label({float(test_x), float(test_y)});
  
  std::cout << test_x << " " << op << " " << test_y << " = ";
  if (resutl == range-1)
    std::cout << "invalid" << std::endl;
  else
    std::cout << (resutl + min_t) << std::endl;
}


static void usage(const char *argv0) {
  std::cout << "Usage: " << argv0 << " --data_path path_to_dataset_folder"
            << " --learning_rate 1"
            << " --epochs 30"
            << " --batch_size 16" << std::endl;
}

int main(int argc, char **argv) {

  double learning_rate                   = 1;
  int epochs                             = 30;
  std::string data_path                  = "";
  int batch_size                         = 16;
  int test_x = -1;
  int test_y = -1;
  char op = ' ';

  if (argc == 2) {
    std::string argname(argv[1]);
    if (argname == "--help" || argname == "-h") {
      usage(argv[0]);
      return 0;
    }
  }
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--batch_size") {
      batch_size = atoi(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else if (argname == "--expr") {
      auto expr = std::string(argv[count + 1]);
      auto op_pos = expr.find_first_not_of("0123456789");
      if (op_pos != std::string::npos) {
        test_x = stoi(expr.substr(0, op_pos));
        test_y = stoi(expr.substr(op_pos+1));
        op = expr[op_pos];
      }

    } else {
      std::cerr << "Invalid parameter specified - \"" << argname << "\""
                << std::endl;
      usage(argv[0]);
      return -1;
    }
  }
  std::cout << "Running with the following parameters:" << std::endl
            << "Data path: " << data_path << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Batch size: " << batch_size << std::endl
            << "Number of epochs: " << epochs << std::endl
            << std::endl;
  try {

    if (op != ' ')
      train_arth(learning_rate, epochs, op, test_x, test_y);
    else 
      train_mnist(data_path, learning_rate, epochs, batch_size);

  } catch (std::exception &err) {
    std::cerr << "Exception: " << err.what() << std::endl;
  }
  return 0;
}
