from multiml.saver import Saver


def test_saver():
    saver = Saver(mode='shelve')
    assert saver._serial_id == 0

    saver.open()
    assert saver._state == 'open'

    saver['key0'] = 'value0'
    assert saver.keys('shelve') == ['key0']
    assert saver['key0'] == 'value0'
    assert saver._shelve['key0'] == 'value0'

    # change shelve to dict
    saver.set_mode('dict')

    saver['key1'] = 'value1'
    assert saver.keys('dict') == ['key1']
    assert saver['key1'] == 'value1'
    assert saver._dict['key1'] == 'value1'

    assert 'key0' not in saver._dict
    assert 'key1' not in saver._shelve

    # save dict objects to shelve
    saver.save()

    assert set(saver.keys()) == set(['key0', 'key1'])
    assert 'key0' not in saver._dict
    assert 'key1' not in saver._dict
    assert 'key0' in saver._shelve
    assert 'key1' in saver._shelve

    # move data from shelve to dict
    saver.to_memory('key0')
    assert 'key0' not in saver._shelve
    assert 'key0' in saver._dict

    # move data from dict to shelve
    saver.to_storage('key0')
    assert 'key0' in saver._shelve
    assert 'key0' not in saver._dict

    saver.set_mode('shelve')
    saver.dump_ml(key='ml0', param0='param0', param1='param1') 
    assert saver.load_ml(key='ml0')['param0'] == 'param0'

    saver.close()
    assert saver._state == 'close'

if __name__ == '__main__':
    test_saver()
