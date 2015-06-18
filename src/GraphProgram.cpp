
enum edge_direction {OUT_EDGES, IN_EDGES, ALL_EDGES};
//enum execution_direction {PUSH, PULL};
enum activity_type {ACTIVE_ONLY, ALL_VERTICES};

template <class T, class U, class V> //T::message_type, U::message_reduction_type, V::vertex_property_type
class GraphProgram {
  protected:
    edge_direction order;
//    execution_direction push_or_pull;
    activity_type activity;
//    bool async;

  public:

  GraphProgram() {
    order = OUT_EDGES;
//    push_or_pull = PUSH;
    activity = ACTIVE_ONLY;
//    async = false;
  }

  edge_direction getOrder() const {
    return order;
  }
  // bool getAsync() const {
    // return async;
  // }
  // execution_direction getPushOrPull() const {
    // return push_or_pull;
  // }
  activity_type getActivity() const {
    return activity;
  }

  virtual void reduce_function(U& v, const U& w) const {
    //v += w;
  }

  virtual void process_message(const T& message, const int edge_val, const V& vertexprop, U& res) const {
    //res = message * edge_val;
  }

  virtual bool send_message(const V& vertexprop, T& message) const {
    //message = (T)vertexprop;
    return true;
  }

  virtual void apply(const U& message_out, V& vertexprop)  {
    //vertexprop = (V)message_out;
  }

  virtual void do_every_iteration(int iteration_number) {
  }

};

//-------------------------------------------------------------------------
