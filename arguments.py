
def getArgs():
    args = {}

    args['split_type'] = 'training' # training , validation, testing

    args['load_training'] = True # Load trained weights from memory

    args['object_id'] = 'id_00' # empty for pre training

    args['audio_features_type'] = 'big_mel' # STFT or Mel
    args['data_folder'] = f'D:/Code/AdvancedAudio/Project/data/serialized/fan/{args["audio_features_type"].lower()}'

    args['load_into_memory'] = False

    # Training hyper parameters:
    args['batch_size'] = 8
    args['lr'] = 1e-06
    args['momentum'] = 0.0
    args['epochs'] = 50

    # Model params
    args['dense'] = True
    args['skip'] = True

    return args
