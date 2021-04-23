import numpy as np

from multiml.storegate import StoreGate

##############################################################################
# user defined parameters

data_id = 'test_storegate'
var_names01 = ('var0', 'var1')
var_names23 = ('var2', 'var3')

data0 = np.linspace(0, 9, 10)
data1 = np.linspace(10, 19, 10)
data2 = np.linspace(20, 29, 10)
data3 = np.linspace(30, 39, 10)

data01 = [data0, data1]
data23 = [data2, data3]

data01 = np.array(data01)
data23 = np.array(data23)

data01_bind = data01.transpose(1, 0)
data23_bind = data23.transpose(1, 0)

##############################################################################


def test_storegate_zarr():
    storegate = StoreGate(backend='hybrid',
                          backend_args={'mode': 'w'},
                          data_id=data_id)

    assert storegate.data_id == data_id
    storegate.set_data_id(data_id)
    assert storegate.data_id == data_id

    phase = (0.8, 0.1, 0.1)

    # add new variables
    storegate.add_data(var_names=var_names01, data=data01_bind, phase=phase)
    storegate.add_data(var_names=var_names23, data=data23_bind, phase=phase)

    assert storegate.get_data_ids() == [data_id]

    # change hybrid mode
    storegate.set_mode('numpy')

    storegate.to_storage(var_names=var_names23, phase='all')
    storegate.to_memory(var_names=var_names23, phase='train')

    # update existing variables and data
    storegate.update_data(var_names=var_names23, data=data23_bind, phase=phase)
    storegate.update_data(var_names=var_names23, data=data23_bind, phase=phase)

    # compile for multiai
    storegate.compile()
    storegate.get_metadata()
    storegate.show_info()

    # update data by auto mode
    storegate.update_data(var_names=var_names23, data=data23_bind, phase='all')
    storegate['all'][var_names23][:] = data23_bind
    storegate.compile()

    # get data
    storegate.get_data(var_names=var_names23, phase='train', index=0)
    storegate.get_data(var_names=var_names23, phase='all')
    storegate['all'][var_names23][:]

    # tests
    total_events = len(data0)
    train_events = total_events * phase[0]
    valid_events = total_events * phase[1]
    test_events = total_events * phase[2]

    assert len(storegate['train']) == total_events * phase[0]
    assert len(storegate['valid']) == total_events * phase[1]
    assert len(storegate['test']) == total_events * phase[2]

    assert var_names23[0] in storegate['train']
    assert var_names23[1] in storegate['train']

    storegate_train = storegate['train']

    assert storegate_train[var_names23][:].shape == (train_events,
                                                     len(var_names23))
    assert storegate_train[var_names23][0].shape == (len(var_names23), )
    assert storegate_train[var_names23[0]][:].shape == (train_events, )
    assert storegate_train[var_names23[0]][0] == 20

    # delete data 
    storegate.delete_data(var_names='var2', phase='train')
    del storegate['train'][['var0', 'var1']]
    assert storegate.get_var_names(phase='train') == ['var3']

    # convert dtype
    storegate.astype(var_names='var3', dtype=np.int)
    storegate.onehot(var_names='var3', num_classes=40)
    storegate.argmax(var_names='var3', axis=1)
    assert storegate['train']['var3'][0] == 30


    # test shuffle with numpy mode
    storegate = StoreGate(backend='numpy', data_id=data_id)
    storegate.add_data(var_names=var_names01,
                       data=data01_bind,
                       phase=phase,
                       shuffle=True)
    storegate.add_data(var_names=var_names23,
                       data=data01_bind,
                       phase=phase,
                       shuffle=True)
    storegate.compile()

    storegate_train = storegate['train']
    assert storegate_train[var_names01][0][0] == storegate_train[var_names23][
        0][0]


if __name__ == '__main__':
    test_storegate_zarr()
