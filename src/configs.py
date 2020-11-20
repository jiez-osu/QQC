def get_config(args):
    conf = {
        # data_params
        'ckptdir': '../checkpoint_qcwqq/',
        'sumdir': '../run_qcwqq/',

        # # vocabulary info -- SQL
        # 'qt_len': 20,
        # 'code_len': 120,
        # 'qt_n_words': 10488,
        # 'code_n_words': 13802,

        # vocabulary info -- Python
        'qt_len': 20,
        'code_len': 120,
        'qt_n_words': 15930,
        'code_n_words': 128538,

        # training_params
        'batch_size': 1024,
        'nb_epoch': 100,
        'optimizer': 'adam',
        'lr': 0.001,
        'valid_every': 1,
        'n_eval': 100,
        'log_every': 50,
        'save_every': 10,
        'patience': 50,
        'reload': 0,  # reload>0, model is reloaded.

        # model_params
        'emb_size': 200,
        # recurrent
        'lstm_dims': 400,  # * 2
        'bow_dropout': 0.25,  # dropout for BOW encoder
        'seqenc_dropout': 0.25,  # dropout for sequence encoder encoder
        'margin': 0.05,
        'code_encoder': 'bilstm',  # bow, bilstm
    }

    return conf
