
def getArgs():
    args = {}

    args['split_type'] = 'testing' # training , validation , testing

    args['object_id'] = '' # empty for pre training

    args['audio_features_type'] = 'mel' # STFT or mel
    args['data_folder'] = 'D:/Code/AdvancedAudio/Project/data/serialized/fan/mel'

    args['load_into_memory'] = True

    # Training hyper parameters:
    args['batch_size'] = 16

    return args


