import numpy as np

class TortillaDataStream:
    """
    A TortillaDataStream represents a temporal stream of values, which can be
    scalars or tensors.
    They first get stored in a buffer, and whenever the buffer length is exceeded beyond a threshold,
    the value in the buffer is pushed into the actual datastream.
    a max_buffer_length of 1 can be used in the case all the values want to be stored
    """

    def __init__(self, name, column_names=False, max_buffer_length=10**10):
        self.name = name
        self.column_names = column_names
        self.max_buffer_length = max_buffer_length

        self.datastream = []

        self.reset_buffer()

    def reset_buffer(self):
        # Important to intialize like this,
        # as we want the buffer to be flexible in picking up the data shape
        # with the first addition to the buffer
        self.buffer_empty=True
        self.buffer = np.array([False])
        self.buffer_length = 0

    def add_to_buffer(self, d):
        if self.buffer_length >= self.max_buffer_length:
            self.flush_buffer()

        if not self.buffer_empty:
            assert type(d) == type(self.buffer)
            assert d.shape == self.buffer.shape
            self.buffer = \
                (self.buffer_length/(self.buffer_length+1))*(self.buffer) \
                + (1.0/(self.buffer_length+1))*d
        else:
            self.buffer = d
            self.buffer_empty = False

        self.buffer_length += 1

    def flush_buffer(self):
        # TODO: Add checks to ensure that the buffer is of the same shape
        #       as the datastream elements
        self.datastream.append(self.buffer)
        self.reset_buffer()

if __name__ == "__main__":
    ds = TortillaDataStream(name="test", column_names=["a", "b", "c"])
    ds.add_to_buffer(np.array([1,2,3]))
    print(ds.buffer)
    assert ds.buffer.all() == np.array([1,2,3]).all()
    ds.add_to_buffer(np.array([1,2,3]))
    print(ds.buffer)
    assert ds.buffer.all() == np.array([1,2,3]).all()
    ds.add_to_buffer(np.array([1,2,3]))
    print(ds.buffer)
    assert ds.buffer.all() == np.array([1,2,3]).all()
    ds.flush_buffer()
    print(ds.datastream)
    assert len(ds.datastream) == 1
    assert ds.datastream[0].all() == np.array([1,2,3]).all()
