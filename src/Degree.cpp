//----------------------------------------------------------------------------
template<class V>
class Degree : public GraphProgram<int, int, V> {
  public:

  Degree() {
    this->order = IN_EDGES;
  }

  bool send_message(const V& vertexprop, int& message) const {
    message = 1;
    return true;
  }

  void process_message(const int& message, const int edge_value, const V& vertexprop, int& result) const {
    result = message;
  }

  void reduce_function(int& a, const int& b) const {
    a += b;
  }

  void apply(const int& message_out, V& vertexprop) {
    vertexprop.degree = message_out; 
  }

};


