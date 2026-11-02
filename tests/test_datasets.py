import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from ml_lab.datasets import DatasetLibrary,dataset_record,combine_datasets

APP=str(Path(__file__).resolve().parents[1]/'app.py')


def record(frame,name):
    return dataset_record(frame.to_csv(index=False).encode(),name)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.first=record(pd.DataFrame({'datetime':pd.date_range('2024-01-01',periods=40,freq='h'),'y':range(40)}),'first.csv')
        self.second=record(pd.DataFrame({'datetime':pd.date_range('2024-01-01 10:00',periods=30,freq='h'),'x':range(30)}),'second.csv')

    def test_save_load_restart_and_deduplication(self):
        with tempfile.TemporaryDirectory() as root:
            library=DatasetLibrary(root)
            self.assertEqual(library.list(),[])
            identifier=library.save(self.first)
            self.assertEqual(library.save(self.first),identifier)
            restarted=DatasetLibrary(root)
            self.assertEqual(len(restarted.list()),1)
            loaded=restarted.load(identifier)
            pd.testing.assert_frame_equal(loaded['frame'],self.first['frame'])
            self.assertEqual(loaded['raw'],self.first['raw'])

    def test_settings_persist_and_safe_filenames(self):
        source=dataset_record(b'x;y\n1,2;3,4\n','../../outside.csv',{'separator':';','decimal':','})
        with tempfile.TemporaryDirectory() as root:
            library=DatasetLibrary(root);key=library.save(source)
            self.assertEqual(library.load(key)['frame'].iloc[0,0],1.2)
            self.assertEqual(len(list(Path(root).iterdir())),2)
            with self.assertRaises(ValueError): library.load('../outside')

    def test_corrupted_content_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            library=DatasetLibrary(root);key=library.save(self.first)
            (Path(root)/(key+'.bin')).write_bytes(b'x,y\n1,2\n')
            with self.assertRaisesRegex(ValueError,'изменено'): library.load(key)

    def test_time_alignment_preserves_primary_and_provenance(self):
        combined,notes=combine_datasets([self.first,self.second])
        self.assertEqual(len(combined),40)
        self.assertEqual(combined.datetime.iloc[0],pd.Timestamp('2024-01-01'))
        first=next(c for c in combined if c.endswith(' · y'))
        second=next(c for c in combined if c.endswith(' · x'))
        self.assertIn('first.csv',first);self.assertIn('second.csv',second)
        self.assertEqual(combined[second].isna().sum(),10)
        self.assertEqual(combined[second].iloc[10],0)
        self.assertTrue(any('30 из 40' in note for note in notes))

    def test_time_duplicates_and_no_overlap_rejected(self):
        duplicate=record(pd.DataFrame({'datetime':['2024-01-01','2024-01-01'],'x':[1,2]}),'dup.csv')
        with self.assertRaisesRegex(ValueError,'повторяются'): combine_datasets([self.first,duplicate])
        separate=record(pd.DataFrame({'datetime':['2025-01-01'],'x':[1]}),'separate.csv')
        with self.assertRaisesRegex(ValueError,'нет общих дат'): combine_datasets([self.first,separate])

    def test_row_alignment_requires_equal_lengths_and_retains_primary_time(self):
        other=record(pd.DataFrame({'z':range(40),'a':range(40)}),'other.csv')
        combined,_=combine_datasets([self.first,other],'rows')
        self.assertEqual(len(combined),40)
        self.assertIn('datetime',combined)
        with self.assertRaisesRegex(ValueError,'количество строк'): combine_datasets([self.first,self.second],'rows')

    def test_single_dataset_is_unchanged_and_not_persisted_implicitly(self):
        combined,_=combine_datasets([self.first])
        pd.testing.assert_frame_equal(combined,self.first['frame'])
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(DatasetLibrary(root).list(),[])

    def test_ui_session_save_and_saved_selection(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('ml_lab.dataset_ui.DatasetLibrary',side_effect=lambda _:DatasetLibrary(root)):
                at=AppTest.from_file(APP).run(timeout=60)
                at.session_state.datasets_session={self.first['id']:self.first,self.second['id']:self.second}
                at.run()
                self.assertFalse(at.exception)
                self.assertEqual(DatasetLibrary(root).list(),[])
                at.button(key='datasets_save_'+self.first['id']).click().run()
                self.assertEqual(len(DatasetLibrary(root).list()),1)
                at.radio(key='datasets_source').set_value('Загруженные и сохранённые').run()
                at.multiselect(key='datasets_selected').set_value([self.first['id'],self.second['id']]).run()
                self.assertFalse(at.exception)
                self.assertFalse(at.error)
                self.assertTrue(any('first.csv' in option for option in at.selectbox(key='cfg_target').options))
                # Restart without the temporary session: the explicitly saved dataset remains available.
                again=AppTest.from_file(APP).run(timeout=60)
                again.radio(key='datasets_source').set_value('Загруженные и сохранённые').run()
                again.multiselect(key='datasets_selected').set_value([self.first['id']]).run()
                self.assertFalse(again.exception)
                self.assertEqual(again.selectbox(key='cfg_target').value,'y')


if __name__=='__main__': unittest.main()
