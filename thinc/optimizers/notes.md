# Torch optimizers in thinc

new registry for torch optimizers
api will stick to torch semantics
will optimize all thinc params
iterate over the trainable pipes and register the params with the torch optimizer
uses pytorch schedulers

pytorch params can be registered as groups and have different lr for each

## Issues

- WE CANT DO ZERO_COPY CONVERSIONS AND INPLACE MODIFICATIONS OF CPU ARRAYS in the thinc optimizer as numpy's dlpack wrapper returns a readonly view

## Architecture

- Increase numpy min version to 1.24 and cupy to 12.0 (so that we always have dlpack support)
- Have a top-level optimizer interface with:

  - `initialize`: takes Language, walks through all models, registers params with it (thinc,torch,tf,etc), misc LR init, etc

    - skip pipes that not trainable/are excluded
    - some option to allow custom parameter groups (like for the different transformer layers)?
    - differentiate between torch tensors that are essentialy wrappers around xp tensors, and those that are true torch tensors (from torch models)
    - what to do about the shims that have their own parameter store?
      - tf/mxnet create an ad-hoc flat repr of all the params (and their grads),pass it to the thinc optimizer, unflatten and set the original backingstores
      - should we just move the registration to `finish_update`/`set_grad`?
        - then we should ensure that the shims allocate a flat tensor at init and reuse to for all gradient updates
        - or we make the registrations ephemeral, ie. they are reset after each step

  - `set_grad`: for a given Model and parameter name, register the gradient for optimization in step

    - how do we handle the grads? do we just create a dlpack view into the grad xp tensor (for thinc models) from the paramserver?
      - if so, how will `zero_grad()` interact with it? multiply with 0 instead?
        - either we call with `set_to_none=False` so the torch grads in wrapped thinc tensors are preserved,
          but this applies to all registered params; means we lose performance for pure torch params
        - alternatively, individually iterate over the tracked params and set them to zero while setting the
          pure torch ones to `None` [FOLLOW THE CODE IN THE TORCH OPTIMIZER'S METHOD]
      - looking at the paramserver code, grad tensors are allocated on demand (and exclusively through `inc_grad`). But they remain stable therePafter (reused for the rest of the session).
      - so, we can prolly get away by registering it just once/param.
      - the alternative is to always allocate grads for all thinc model parameters preemptively (not good memory-wise)
    - pass an additional callback that receives the updated parameters, used to update the backstore
      - needed for the thinc optimizer to keep to from having to directly invoke sideeffects on models/shims
      - torch optimizer can just ignore this callback as the torch tensor wrappers

  - `step`: perform the optimization pass, update registered params, zero out grads, update LR
    - how does gradient scaling factor into this?

- Model:
  - `finish_update`: walks through paramters and calls `optimizer.set_grad`

### Custom optimizer params:

- config has something like:
  ```
  [training.optimizer]
  per_param_lr = {
    "transformer1.<param_name>": { "@schedules": "SomeSchedule.v1", ...}
    ...
  }
  ```
  - This will require modifications to the config to support the resolution/
    construction of nested entities.
- ultimately, we want something that will let us uniquely identify a parameter
  in a given component's model
- During `init_optimizer()`, we map the tags from above to the (int, str) keys pertaining to the component models that we use internally
  - this will only work for (thinc) params that have been initialized in `Model.initialize`; any params added during runtime won't be supported
  - Sure, this does seem a bit brittle but it's the only way (I can think of right now) that allows us to support different optimizer options for the different instances of the same parameter, e.g: the layers of different transformers in a pipeline that doesn't share a tok2vec.
  - we could, of course, not support that distinction to keep things simple. In that case we can simply stick to a string tag and compare it to the string part of the internal key
- We instantiate and store the different options
- During param registration, we get the key of parameter which includes a string tag. This will be used to match the parameter with its custom options in `optimizer.step`
  - using a `with` context manager

### Torch LR schedulers

- As such, PyTorch LR schedulers cannot be used on a param group basis. This is because they are
  applied after each optimizer update on all param groups at once, i.e., the initial LR of each
  param group is mutated with the scheduler's logic and updated in-place.
  - Since we want per-parameter lr schedules, we need our own Pytorch scheduler (wrapper) that
    looks up the individual schedulers and applies them.
  - Also, the torch schedulers are meant to be used on a per-epoch basis (as opposed to per-step)
- Furthermore, we can only apply schedules to the LR with the torch schedulers. Thinc, however, currently
  supports schedules for all optimizer options.
- Therefore, it makes sense to continue using the thinc schedule implementations for the torch optimizer too
