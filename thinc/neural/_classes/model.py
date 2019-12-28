import numpy
import contextlib
import srsly
import threading

from .. import util
from ..ops import NumpyOps, CupyOps
from ..mem import Memory
from ..util import get_ops, copy_array


class Model(object):
    """Base class for Thinc models and layers."""

    name = "model"
    id = 0
    ops = NumpyOps()
    Ops = NumpyOps
    descriptions = []
    _thread_local = threading.local()

    @classmethod
    @contextlib.contextmanager
    def define_operators(cls, operators):
        """Bind operators to specified functions for the scope of the context:

        Example
        -------

            model = Model()
            other = Model()
            with Model.define_operators({"+": lambda self, other: "plus"}):
                print(model + other)
                # "plus"
            print(model + other)
            # Raises TypeError --- binding limited to scope of with block.
        """
        curr_operators = dict(getattr(cls._thread_local, "operators", {}))
        cls._thread_local.operators = dict(operators)
        yield
        cls._thread_local.operators = dict(curr_operators)

    @classmethod
    @contextlib.contextmanager
    def use_device(cls, device):
        """Change the device to execute on for the scope of the block."""
        if device == cls.ops.device:
            yield
        else:
            curr_Ops = cls.Ops
            curr_ops = cls.ops
            cls.Ops = get_ops(device)
            cls.ops = cls.Ops()
            yield
            cls.Ops = curr_Ops
            cls.ops = curr_ops

    def __init__(self, name=None, ops=None, layers=None):
        self.descriptions = dict(self.__class__.descriptions)
        self.name = self.__class__.name if name is None else name
        if ops is None:
            self.Ops = self.__class__.Ops
            self.ops = self.Ops()
        else:
            self.Ops = ops.__class__
            self.ops = ops
        self.on_data_hooks = []
        self._mem = Memory(self.ops)
        self._params = {}
        self._dims = {}
        self._grads = {}
        self._layers = [] if layers is None else list(layers)
        self.on_data_hooks.append(lambda model, X, Y=None: model.infer_dimensions(X, Y))

        for attr, install in self.descriptions.items():
            install(attr, self)
        self.set_id()

    def __getstate__(self):
        return srsly.pickle_dumps(self.__dict__)

    def __setstate__(self, state_data):
        self.__dict__ = srsly.pickle_loads(state_data)

    def add_layer(self, layer):
        """Add a child layer to the model."""
        self._layers.append(layer)
    
    def dim_is_unset(self, name):
        return self.has_dim(name) and self.get_dim(name) is None

    def has_dim(self, name):
        """Check whether the model has a dimension of a given name."""
        return name in self._dims

    def get_dim(self, name):
        """Retrieve the value of a dimension of the given name, or None if unset."""
        return self._dims.get(name, None)

    def set_dim(self, name, value):
        """Set a value for a dimension."""
        self._dims[name] = value

    def has_param(self, name):
        """Check whether the model has a weights parameter of the given name."""
        return name in self._params

    def get_param(self, name):
        """Retrieve a weights parameter by name."""
        key = (self.id, name)
        if key in self._mem:
            return self._mem[key]
        else:
            param_info = self._params[name]
            shape = param_info.get_shape(self)
            if any(dim is None for dim in shape):
                raise ValueError(f"Dimensions unset!: {shape}")
            data = self._mem.add(key, shape)
            if param_info.init is not None:
                param_info.init(data, self.ops)
            return data

    def set_param(self, name, value):
        """Set a weights parameter's value."""
        data = self._mem.get((self.id, name))
        copy_array(dst=data, src=value)

    def has_grad(self, name):
        """Check whether the model has a gradient of the given name."""
        return name in self._grads

    def get_grad(self, name):
        """Get a gradient from the model."""
        key = (self.id, name)
        if key in self._mem:
            return self._mem.get(key)
        else:
            grad_info = self._grads[name]
            param_key = (self.id, grad_info.param_name)
            grad = self._mem.add_gradient(key, param_key)
            return grad

    def set_grad(self, name, value):
        """Set a gradient value for the model."""
        data = self._mem.get((self.id, name))
        copy_array(dst=data, src=value)

    def set_id(self):
        """Update the model's ID, and also update the ID recursively for children."""
        Model.id += 1
        self.id = Model.id
        for layer in self._layers:
            layer.set_id()

    def begin_training(self, X=None, Y=None):
        """Lifecycle method that can be called to initiate training. Triggers
        calls to any functions registered in model.on_data_hooks.

        You can provide example input and output data in the X and Y arguments.
        This is mostly useful for allowing shapes to be inferred from the example
        data.
        """
        for hook in self.on_data_hooks:
            hook(self, X=X, Y=Y)

    def infer_dimensions(self, X=None, Y=None):
        """Infer missing dimensions from example data."""
        if X is not None and self.dim_is_unset("nI"):
            self.set_dim("nI", util.get_width(X))
        if Y is not None and self.dim_is_unset("nO"):
            self.set_dim("nO", util.get_width(Y))

    def begin_update(self, X):
        """Run the model over a batch of data, returning the output and a callback
        to complete the backward pass.

        X: A batch of input data.

        RETURNS:
            A tuple (Y, finish_update), where Y is a batch of output data,
            and finish_update is a callback that takes the gradient with
            respect to the output and an optimizer function, and returns
            the gradient with respect to the input.
        """
        raise NotImplementedError

    def finish_update(self, optimizer):
        """Update parameters with current gradients.
        
        optimizer (Callable[array, array, key=None]):
            The optimizer. The function is called with each parameter and
            gradient of the model.
        """
        optimizer(self._mem.weights, self._mem.gradient, key=self.id)
        seen = set([self.id])
        for node in self.walk():
            if node.id not in seen:
                node.finish_update(optimizer)
                seen.add(node.id)

    def predict(self, X):
        self.disable_dropout()
        y, _ = self.begin_update(X)
        self.enable_dropout()
        return y

    def set_dropout(self, rate):
        for node in self.walk():
            if node.name == "dropout":
                node.drop = rate

    def disable_dropout(self):
        for node in self.walk():
            if node.name == "dropout":
                node.is_enabled = False
    
    def enable_dropout(self):
        for node in self.walk():
            if node.name == "dropout":
                node.is_enabled = True
 
    def __call__(self, x):
        # I think we should remove this.
        return self.predict(x)
 
    @contextlib.contextmanager
    def use_params(self, params):  # pragma: no cover
        """Context manager to temporarily set the model's parameters to specified
        values.

        params (dict): A dictionary keyed by model IDs, whose values are arrays
            of weight values.
        """
        backup = None
        weights = self._mem.weights
        if self.id in params:
            param = params[self.id]
            backup = weights.copy()
            copy_array(dst=weights, src=param)
        if hasattr(self, "_layers"):
            contexts = [layer.use_params(params) for layer in self._layers]
            for context in contexts:
                next(context.gen)
        yield
        if backup is not None:
            copy_array(dst=self._mem.weights, src=backup)
        for i, context in enumerate(contexts):
            # This is ridiculous, but apparently it's what you
            # have to do to make this work across Python 2/3?
            try:
                next(context.gen)
            except StopIteration:
                pass
     
    def walk(self):
        """Iterate out layers of the model, breadth-first."""
        queue = [self]
        seen = set()
        for node in queue:
            if node.id in seen:
                continue
            seen.add(node.id)
            yield node
            if hasattr(node, "_layers"):
                queue.extend(node._layers)

    def get_gradients(self):
        """Get non-zero gradients of the model's parameters, as a dictionary
        keyed by the parameter ID. The values are (weights, gradients) tuples.
        """
        gradients = {}
        for node in self.walk():
            if hasattr(node, "_mem") and node._mem.gradient.any():
                gradients[node.id] = [node._mem.weights, node._mem.gradient]
        return gradients

    def to_gpu(self, device_num):
        """Transfer the model to a given GPU device."""
        import cupy.cuda.device

        device = cupy.cuda.device.Device(device_num)
        device.use()
        queue = [self]
        for layer in queue:
            layer.ops = CupyOps()
            layer.Ops = CupyOps
            if hasattr(layer, "_mem"):
                layer._mem._mem = self.ops.xp.asarray(layer._mem._mem)
                layer._mem.ops = layer.ops
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)
        return device

    def to_cpu(self):
        """Copy the model to CPU."""
        queue = [self]
        for layer in queue:
            layer.ops = NumpyOps()
            layer.Ops = NumpyOps
            if hasattr(layer, "_mem"):
                if hasattr(layer._mem._mem, "get"):
                    layer._mem._mem = layer._mem._mem.get()
                layer._mem.ops = layer.ops
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)

    def to_bytes(self):
        """Serialize the model to a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        weights = []
        queue = [self]
        i = 0
        for layer in queue:
            # Hack to support saving/loading PyTorch models. TODO: Improve
            if hasattr(layer, "_model") and not isinstance(layer._model, Model):
                weights.append(layer.to_bytes())
            elif hasattr(layer, "_mem"):
                weights.append(
                    {b"dims": dict(sorted(layer._dims.items())), b"params": []}
                )
                if hasattr(layer, "seed"):
                    weights[-1][b"seed"] = layer.seed

                offsets = sorted(layer._mem._offsets.items())
                for (id_, name), (start, row, shape) in offsets:
                    if row == 1:
                        continue
                    param = layer._mem.get((id_, name))
                    if not isinstance(layer._mem.weights, numpy.ndarray):
                        param = param.get()
                    weights[-1][b"params"].append(
                        {
                            b"name": name,
                            b"offset": start,
                            b"shape": shape,
                            b"value": param,
                        }
                    )
                i += 1
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)
        return srsly.msgpack_dumps({b"weights": weights})

    def from_bytes(self, bytes_data):
        """Deserialize the model from a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        data = srsly.msgpack_loads(bytes_data)
        weights = data[b"weights"]
        queue = [self]
        i = 0
        for layer in queue:
            # Hack to support saving/loading PyTorch models. TODO: Improve
            if hasattr(layer, "_model") and not isinstance(layer._model, Model):
                layer.from_bytes(weights[i])
                i += 1
            elif hasattr(layer, "_mem"):
                if b"seed" in weights[i]:
                    layer.seed = weights[i][b"seed"]
                for dim, value in weights[i][b"dims"].items():
                    if isinstance(dim, bytes):
                        dim = dim.decode("utf8")
                    setattr(layer, dim, value)
                for param in weights[i][b"params"]:
                    name = param[b"name"]
                    if isinstance(name, bytes):
                        name = name.decode("utf8")
                    dest = getattr(layer, name)
                    copy_array(dst=dest, src=param[b"value"])
                i += 1
            if hasattr(layer, "_layers"):
                queue.extend(layer._layers)
        return self

    def to_disk(self, path):
        """Serialize the model to disk. Most models will serialize to a single
        file, which should just be the bytes contents of model.to_bytes().
        """
        path = util.ensure_path(path)
        with path.open("wb") as file_:
            file_.write(self.to_bytes())

    def from_disk(self, path):
        """Deserialize the model from disk. Most models will serialize to a single
        file, which should just be the bytes contents of model.to_bytes().
        """
        path = util.ensure_path(path)
        with path.open("rb") as file_:
            bytes_data = file_.read()
        return self.from_bytes(bytes_data)

    def __add__(self, other):
        """Apply the function bound to the '+' operator."""
        if "+" not in self._thread_local.operators:
            raise TypeError("Undefined operator: +")
        return self._thread_local.operators["+"](self, other)

    def __sub__(self, other):
        """Apply the function bound to the '-' operator."""
        if "-" not in self._thread_local.operators:
            raise TypeError("Undefined operator: -")
        return self._thread_local.operators["-"](self, other)

    def __mul__(self, other):
        """Apply the function bound to the '*' operator."""
        if "*" not in self._thread_local.operators:
            raise TypeError("Undefined operator: *")
        return self._thread_local.operators["*"](self, other)

    def __matmul__(self, other):
        """Apply the function bound to the '@' operator."""
        if "@" not in self._thread_local.operators:
            raise TypeError("Undefined operator: @")
        return self._thread_local.operators["@"](self, other)

    def __div__(self, other):
        """Apply the function bound to the '/' operator."""
        if "/" not in self._thread_local.operators:
            raise TypeError("Undefined operator: /")
        return self._thread_local.operators["/"](self, other)

    def __truediv__(self, other):  # pragma: no cover
        """Apply the function bound to the '/' operator."""
        if "/" not in self._thread_local.operators:
            raise TypeError("Undefined operator: /")
        return self._thread_local.operators["/"](self, other)

    def __floordiv__(self, other):
        """Apply the function bound to the '//' operator."""
        if "//" not in self._thread_local.operators:
            raise TypeError("Undefined operator: //")
        return self._thread_local.operators["//"](self, other)

    def __mod__(self, other):
        """Apply the function bound to the '%' operator."""
        if "%" not in self._thread_local.operators:
            raise TypeError("Undefined operator: %")
        return self._thread_local.operators["%"](self, other)

    def __pow__(self, other, modulo=None):
        """Apply the function bound to the '**' operator."""
        if "**" not in self._thread_local.operators:
            raise TypeError("Undefined operator: **")
        return self._thread_local.operators["**"](self, other)

    def __lshift__(self, other):
        """Apply the function bound to the '<<' operator."""
        if "<<" not in self._thread_local.operators:
            raise TypeError("Undefined operator: <<")
        return self._thread_local.operators["<<"](self, other)

    def __rshift__(self, other):
        """Apply the function bound to the '>>' operator."""
        if ">>" not in self._thread_local.operators:
            raise TypeError("Undefined operator: >>")
        return self._thread_local.operators[">>"](self, other)

    def __and__(self, other):
        """Apply the function bound to the '&' operator."""
        if "&" not in self._thread_local.operators:
            raise TypeError("Undefined operator: &")
        return self._thread_local.operators["&"](self, other)

    def __xor__(self, other):
        """Apply the function bound to the '^' operator."""
        if "^" not in self._thread_local.operators:
            raise TypeError("Undefined operator: ^")
        return self._thread_local.operators["^"](self, other)

    def __or__(self, other):
        """Apply the function bound to the '|' operator."""
        if "|" not in self._thread_local.operators:
            raise TypeError("Undefined operator: |")
        return self._thread_local.operators["|"](self, other)
