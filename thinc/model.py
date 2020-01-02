from typing import Dict, List, Callable, Optional, Any, Union, Iterable, Set
from typing import Sequence, Tuple, TypeVar
import numpy
import contextlib
import srsly
from pathlib import Path
import copy

from .backends import NumpyOps, CupyOps, get_current_ops
from .optimizers import Optimizer  # noqa: F401
from .backends.mem import Memory
from .shims import Shim
from .util import copy_array, get_width, create_thread_local
from .types import Array

InT = TypeVar("InT")
OutT = TypeVar("OutT")


def create_init(initializers: Dict[str, Callable]) -> Callable:
    """Create an init function, given a dictionary of parameter initializers."""

    def init(
        model: Model, X: Optional[Array] = None, Y: Optional[Array] = None
    ) -> None:
        if X is not None:
            model.set_dim("nI", get_width(X))
        if Y is not None:
            model.set_dim("nO", get_width(Y))
        W = model.ops.allocate((model.get_dim("nO"), model.get_dim("nI")))
        b = model.ops.allocate((model.get_dim("nO"),))
        if "W" in initializers:
            initializers["W"](W, inplace=True)
        if "b" in initializers:
            initializers["b"](b, inplace=True)
        model.set_param("W", W)
        model.set_param("b", b)

    return init


class Model:
    """Class for implementing Thinc models and layers."""

    global_id: int = 0
    _thread_local = create_thread_local({"operators": {}})

    name: str
    ops: Union[NumpyOps, CupyOps]
    id: int
    _func: Callable
    _init: Callable
    _mem: Memory
    _params: Dict[str, Optional[bool]]
    _dims: Dict[str, Optional[int]]
    _grads: Dict[str, Optional[bool]]
    _layers: List["Model"]
    _shims: List[Shim]
    _attrs: Dict[str, Any]

    # This "locks" the class, so we get an error if you try to assign to
    # an unexpected variable.
    __slots__ = [
        "name",
        "id",
        "ops",
        "_func",
        "_init",
        "_mem",
        "_params",
        "_dims",
        "_grads",
        "_attrs",
        "_layers",
        "_shims",
    ]

    def __init__(
        self,
        name: str,
        forward: Callable,
        *,
        init: Callable = lambda *a, **k: None,
        dims: Dict[str, Optional[int]] = {},
        params: Dict[str, Optional[bool]] = {},
        grads: Dict[str, Optional[Array]] = {},
        layers: Sequence["Model"] = [],
        shims: List[Shim] = [],
        attrs: Dict[str, object] = {},
        ops: Optional[Union[NumpyOps, CupyOps]] = None,
    ):
        self.name = name
        # Assign to callable attrs: https://github.com/python/mypy/issues/2427
        setattr(self, "_func", forward)
        setattr(self, "_init", init)
        self.ops = ops if ops is not None else get_current_ops()
        self._mem = Memory(self.ops)
        self._dims = dict(dims)
        self._attrs = dict(attrs)
        self._layers = list(layers)
        self._shims = list(shims)
        self.__class__.global_id += 1
        self.id = self.__class__.global_id
        self._params = {}
        self._grads = {}
        for name, value in params.items():
            self._params[name] = None
            if value is not None:
                self.set_param(name, value)
        for name, value in grads.items():
            self._grads[name] = None
            if value is not None:
                self.set_grad(name, value)

    @property
    def layers(self):
        return self._layers

    @property
    def shims(self):
        return self._shims

    @classmethod
    @contextlib.contextmanager
    def define_operators(cls, operators):
        """Bind operators to specified functions for the scope of the context:

        Example:
            model = Model()
            other = Model()
            with Model.define_operators({"+": lambda self, other: "plus"}):
                print(model + other)
                # "plus"
            print(model + other)
            # Raises TypeError --- binding limited to scope of with block.
        """
        curr_operators = dict(cls._thread_local.operators)
        cls._thread_local.operators = dict(operators)
        yield
        cls._thread_local.operators = dict(curr_operators)

    def has_dim(self, name: str) -> Optional[bool]:
        """Check whether the model has a dimension of a given name. If the
        dimension is registered but the value is unset, returns None. 
        """
        if name not in self._dims:
            return False
        elif self._dims[name] is not None:
            return True
        else:
            return None

    def get_dim(self, name: str) -> int:
        """Retrieve the value of a dimension of the given name, or None if unset."""
        if name not in self._dims:
            raise KeyError(f"Can't get dimension '{name}'")
        value = self._dims[name]
        if value is None:
            raise ValueError(f"Cannot get dimension '{name}': value unset.")
        else:
            return value

    def set_dim(self, name: str, value: int) -> None:
        """Set a value for a dimension."""
        if name not in self._dims:
            raise KeyError(f"Can't set dimension '{name}'")
        self._dims[name] = value

    def has_param(self, name: str) -> Optional[bool]:
        """Check whether the model has a weights parameter of the given name.

        Returns None if the parameter is registered but currently unset. 
        """
        if name not in self._params:
            return False
        elif self._params[name] is not None:
            return True
        else:
            return None

    def get_param(self, name: str) -> Array:
        """Retrieve a weights parameter by name."""
        if name not in self._params:
            raise KeyError(f"Unknown param: {name}")
        key = (self.id, name)
        if key not in self._mem:
            raise KeyError(f"Parameter '{name}' as not been allocated yet")
        return self._mem[key]

    def set_param(self, name: str, value: Array) -> None:
        """Set a weights parameter's value."""
        key = (self.id, name)
        if key not in self._mem:
            self._mem.add(key, value.shape)
        data = self._mem.get((self.id, name))
        copy_array(dst=data, src=value)
        self._params[name] = True

    def inc_grad(self, name: str, value: Array) -> None:
        """Check whether the model has a gradient of the given name."""
        grad_name = f"d_{name}"
        key = (self.id, grad_name)
        param_key = (self.id, name)
        if key in self._mem:
            grad = self._mem.get(key)
        else:
            grad = self._mem.add_gradient(key, param_key)
        grad += value
        self._grads[grad_name] = True

    def has_grad(self, name: str) -> Optional[bool]:
        """Check whether the model has a non-zero gradient for a parameter.
        Returns None if the gradient is allocated but currently 0. 
        """
        grad_name = f"d_{name}"
        key = (self.id, grad_name)
        if key not in self._mem:
            return False
        elif not self._mem[key].any():
            return None
        else:
            return True

    def get_grad(self, name: str) -> Array:
        """Get a gradient from the model."""
        grad_name = f"d_{name}"
        key = (self.id, grad_name)
        if key not in self._mem:
            raise KeyError(f"Gradient '{grad_name}' as not been allocated yet")
        return self._mem[key]

    def set_grad(self, name: str, value: Array) -> None:
        """Set a gradient value for the model."""
        grad_name = f"d_{name}"
        data = self._mem.get((self.id, grad_name))
        copy_array(dst=data, src=value)

    def has_attr(self, name: str) -> bool:
        """Check whether the model has the given attribute."""
        return name in self._attrs

    def get_attr(self, name: str) -> Any:
        """Get the attribute. Raises KeyError if not present."""
        if name not in self._attrs:
            raise KeyError(f"Can't get attribute '{name}'")
        return self._attrs[name]

    def set_attr(self, name: str, value: Any) -> None:
        """Set the attribute to the given value."""
        self._attrs[name] = value

    def __call__(self, X: Any, is_train: bool = False) -> Tuple[Any, Callable]:
        return self._func(self, X, is_train=is_train)

    def initialize(self, X: Optional[Any] = None, Y: Optional[Any] = None) -> "Model":
        if self._init is not None:
            self._init(self, X=X, Y=Y)
        return self

    def begin_update(self, X: InT) -> Tuple[OutT, Callable[[InT], OutT]]:
        """Run the model over a batch of data, returning the output and a callback
        to complete the backward pass.

        X: A batch of input data.

        RETURNS:
            A tuple (Y, finish_update), where Y is a batch of output data,
            and finish_update is a callback that takes the gradient with
            respect to the output and an optimizer function, and returns
            the gradient with respect to the input.
        """
        return self._func(self, X, is_train=True)

    def predict(self, X: Any) -> Any:
        return self._func(self, X, is_train=False)[0]

    def finish_update(self, optimizer: Optimizer) -> None:
        """Update parameters with current gradients.

        optimizer (Callable[array, array, key=None]):
            The optimizer. The function is called with each parameter and
            gradient of the model.
        """
        optimizer(self._mem.weights, self._mem.gradient, key=self.id)
        for shim in self.shims:
            shim.finish_update(optimizer)
        seen = set([self.id])
        for node in self.walk():
            if node.id not in seen:
                node.finish_update(optimizer)
                seen.add(node.id)
                for shim in node.shims:
                    shim.finish_update(optimizer)

    @contextlib.contextmanager
    def use_params(self, params: Dict[int, Array]):
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
        contexts = []
        for layer in self.layers:
            contexts.append(next(layer.use_params(params).gen))
        for shim in self.shims:
            contexts.append(next(shim.use_params(params).gen))
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

    def walk(self) -> Iterable["Model"]:
        """Iterate out layers of the model, breadth-first."""
        queue = [self]
        seen: Set[int] = set()
        for node in queue:
            if id(node) in seen:
                continue
            seen.add(id(node))
            yield node
            queue.extend(node.layers)

    def get_gradients(self) -> Dict[int, Array]:
        """Get non-zero gradients of the model's parameters, as a dictionary
        keyed by the parameter ID. The values are (weights, gradients) tuples.
        """
        gradients = {}
        for node in self.walk():
            if hasattr(node, "_mem") and node._mem.gradient.any():
                gradients[node.id] = [node._mem.weights, node._mem.gradient]
        return gradients

    def copy(self) -> "Model":
        """
        Create a copy of the model, its attributes, and its parameters. Any child
        layers will also be deep-copied. The copy will receive a distinct `model.id`
        value.
        """
        copied = Model(
            self.name,
            self._func,
            init=self._init,
            params=copy.deepcopy(self._params),
            grads=copy.deepcopy(self._grads),
            dims=copy.deepcopy(self._dims),
            attrs=copy.deepcopy(self._attrs),
            layers=[layer.copy() for layer in self.layers],
        )
        # The `_params` and `_grads` dicts don't hold the actual values --
        # those are within the `model._mem` object. So we need to call `set_param`
        # on the copy.
        for name, is_allocated in self._params.items():
            if is_allocated:
                copied.set_param(name, self.get_param(name))
        for name, is_allocated in self._grads.items():
            if is_allocated:
                copied.set_grad(name, self.get_grad(name))
        return copied

    def to_gpu(self, device_num: int) -> None:
        """Transfer the model to a given GPU device."""
        import cupy.cuda.device

        device = cupy.cuda.device.Device(device_num)
        device.use()
        for layer in self.walk():
            layer.ops = CupyOps()
            if hasattr(layer, "_mem"):
                layer._mem._mem = self.ops.xp.asarray(layer._mem._mem)
                layer._mem.ops = layer.ops
        return device

    def to_cpu(self) -> None:
        """Copy the model to CPU."""
        for layer in self.walk():
            layer.ops = NumpyOps()
            if hasattr(layer, "_mem"):
                if hasattr(layer._mem._mem, "get"):
                    layer._mem._mem = layer._mem._mem.get()
                layer._mem.ops = layer.ops

    def to_bytes(self) -> bytes:
        """Serialize the model to a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        weights: List[Union[str, Dict[str, Any]]] = []
        nodes = list(self.walk())
        for i, layer in enumerate(nodes):
            # Separate attrs that need to be serialized/deserialized with
            # to_/from_bytes.
            obj_attrs = {}
            flat_attrs = {}
            for name, value in layer._attrs.items():
                if hasattr(value, "to_bytes"):
                    obj_attrs[name] = value.to_bytes()
                else:
                    flat_attrs[name] = value
            weights.append(
                {
                    "dims": layer._dims,
                    "params": [],
                    "obj_attrs": obj_attrs,
                    "flat_attrs": flat_attrs,
                }
            )
            for (id_, name), (start, row, shape) in layer._mem._offsets.items():
                if row == 1:
                    continue
                param = layer._mem.get((id_, name))
                if not isinstance(layer._mem.weights, numpy.ndarray):
                    param = param.get()
                weights[-1]["params"].append(  # type: ignore
                    {"name": name, "offset": start, "shape": shape, "value": param}
                )
        return srsly.msgpack_dumps({"weights": weights})

    def from_bytes(self, bytes_data: bytes) -> "Model":
        """Deserialize the model from a bytes representation. Models are usually
        serialized using msgpack, so you should be able to call msgpack.loads()
        on the data and get back a dictionary with the contents.

        Serialization should round-trip identically, i.e. the same bytes should
        result from loading and serializing a model.
        """
        msg = srsly.msgpack_loads(bytes_data)
        nodes = list(self.walk())
        if len(msg["weights"]) != len(nodes):
            raise ValueError("Cannot deserialize model: mismatched structure.")
        for layer, data in zip(nodes, msg["weights"]):
            for attr, value in data["flat_attrs"].items():
                layer.set_attr(attr, value)
            for attr, value in data["obj_attrs"].items():
                layer.get_attr(attr).from_bytes(value)
            for dim, value in data["dims"].items():
                layer.set_dim(dim, value)
            for param in data["params"]:
                layer.set_param(param["name"], param["value"])
        return self

    def to_disk(self, path: Union[Path, str]) -> None:
        """Serialize the model to disk. Most models will serialize to a single
        file, which should just be the bytes contents of model.to_bytes().
        """
        path = Path(path)
        with path.open("wb") as file_:
            file_.write(self.to_bytes())

    def from_disk(self, path: Union[Path, str]) -> "Model":
        """Deserialize the model from disk. Most models will serialize to a single
        file, which should just be the bytes contents of model.to_bytes().
        """
        path = Path(path)
        with path.open("rb") as file_:
            bytes_data = file_.read()
        return self.from_bytes(bytes_data)

    def __add__(self, other: Any) -> "Model":
        """Apply the function bound to the '+' operator."""
        if "+" not in self._thread_local.operators:
            raise TypeError("Undefined operator: +")
        return self._thread_local.operators["+"](self, other)

    def __sub__(self, other: Any) -> "Model":
        """Apply the function bound to the '-' operator."""
        if "-" not in self._thread_local.operators:
            raise TypeError("Undefined operator: -")
        return self._thread_local.operators["-"](self, other)

    def __mul__(self, other: Any) -> "Model":
        """Apply the function bound to the '*' operator."""
        if "*" not in self._thread_local.operators:
            raise TypeError("Undefined operator: *")
        return self._thread_local.operators["*"](self, other)

    def __matmul__(self, other: Any) -> "Model":
        """Apply the function bound to the '@' operator."""
        if "@" not in self._thread_local.operators:
            raise TypeError("Undefined operator: @")
        return self._thread_local.operators["@"](self, other)

    def __div__(self, other: Any) -> "Model":
        """Apply the function bound to the '/' operator."""
        if "/" not in self._thread_local.operators:
            raise TypeError("Undefined operator: /")
        return self._thread_local.operators["/"](self, other)

    def __truediv__(self, other: Any) -> "Model":
        """Apply the function bound to the '/' operator."""
        if "/" not in self._thread_local.operators:
            raise TypeError("Undefined operator: /")
        return self._thread_local.operators["/"](self, other)

    def __floordiv__(self, other: Any) -> "Model":
        """Apply the function bound to the '//' operator."""
        if "//" not in self._thread_local.operators:
            raise TypeError("Undefined operator: //")
        return self._thread_local.operators["//"](self, other)

    def __mod__(self, other: Any) -> "Model":
        """Apply the function bound to the '%' operator."""
        if "%" not in self._thread_local.operators:
            raise TypeError("Undefined operator: %")
        return self._thread_local.operators["%"](self, other)

    def __pow__(self, other: Any, **kwargs) -> "Model":
        """Apply the function bound to the '**' operator."""
        if "**" not in self._thread_local.operators:
            raise TypeError("Undefined operator: **")
        return self._thread_local.operators["**"](self, other)

    def __lshift__(self, other: Any) -> "Model":
        """Apply the function bound to the '<<' operator."""
        if "<<" not in self._thread_local.operators:
            raise TypeError("Undefined operator: <<")
        return self._thread_local.operators["<<"](self, other)

    def __rshift__(self, other: Any) -> "Model":
        """Apply the function bound to the '>>' operator."""
        if ">>" not in self._thread_local.operators:
            raise TypeError("Undefined operator: >>")
        return self._thread_local.operators[">>"](self, other)

    def __and__(self, other: Any) -> "Model":
        """Apply the function bound to the '&' operator."""
        if "&" not in self._thread_local.operators:
            raise TypeError("Undefined operator: &")
        return self._thread_local.operators["&"](self, other)

    def __xor__(self, other: Any) -> "Model":
        """Apply the function bound to the '^' operator."""
        if "^" not in self._thread_local.operators:
            raise TypeError("Undefined operator: ^")
        return self._thread_local.operators["^"](self, other)

    def __or__(self, other: Any) -> "Model":
        """Apply the function bound to the '|' operator."""
        if "|" not in self._thread_local.operators:
            raise TypeError("Undefined operator: |")
        return self._thread_local.operators["|"](self, other)