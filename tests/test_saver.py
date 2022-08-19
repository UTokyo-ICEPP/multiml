from multiml.saver import Saver


def test_saver():
    saver = Saver(mode='zarr')
    assert saver._serial_id == 0

    import numpy as np
    saver['test'] = [0.8387154936790466, 0.8381235599517822]
    saver.delete('test')

    saver['key0'] = 'value0'
    assert saver.keys('zarr') == ['key0']
    assert saver['key0'] == 'value0'
    assert saver._zarr['key0'] == 'value0'

    # change zarr to dict
    saver.set_mode('dict')

    saver['key1'] = 'value1'
    assert saver.keys('dict') == ['key1']
    assert saver['key1'] == 'value1'
    assert saver._dict['key1'] == 'value1'

    assert 'key0' not in saver._dict
    assert 'key1' not in saver._zarr

    # save dict objects to zarr
    saver.save()

    assert set(saver.keys()) == set(['key0', 'key1'])
    assert 'key0' not in saver._dict
    assert 'key1' not in saver._dict
    assert 'key0' in saver._zarr
    assert 'key1' in saver._zarr

    # move data from zarr to dict
    saver.to_memory('key0')
    assert 'key0' not in saver._zarr
    assert 'key0' in saver._dict

    # move data from dict to zarr
    saver.to_storage('key0')
    assert 'key0' in saver._zarr
    assert 'key0' not in saver._dict

    saver.set_mode('zarr')
    saver.dump_ml(key='ml0', param0='param0', param1='param1') 
    assert saver.load_ml(key='ml0')['param0'] == 'param0'

if __name__ == '__main__':
    test_saver()
