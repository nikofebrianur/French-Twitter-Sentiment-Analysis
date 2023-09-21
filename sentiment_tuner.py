
def tuner_fn(fn_args):
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args used to tune models as name/value pairs.
 
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  # Memuat training dan validation dataset yang telah di-preprocessing
  tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
  train_set = input_fn(fn_args.train_files[0], tf_transform_output)
  val_set = input_fn(fn_args.eval_files[0], tf_transform_output)
 
  # Mendefinisikan strategi hyperparameter tuning
  tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=fn_args.working_dir,
                     project_name='kt_hyperband')
 
  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={ 
          "callbacks":[stop_early],
          'x': train_set,
          'validation_data': val_set,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      }
  )
