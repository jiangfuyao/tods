import unittest

from d3m import container, utils
from d3m.metadata import base as metadata_base

from tods.feature_analysis import StatisticalMeanAbs

class StatisticalMeanAbsTestCase(unittest.TestCase):
    def test_basic(self):
        self.maxDiff=None
        main = container.DataFrame({'timestamp': [1, 3, 2, 5], 'values': [1.0, 2.0, 3.0, 4.0], 'b': [1.0, 4.0, 5.0, 6.0]},
                                   columns=['timestamp', 'values', 'b'],
                                   generate_metadata=True)



        self.assertEqual(utils.to_json_structure(main.metadata.to_internal_simple_structure()), [{
            'selector': [],
            'metadata': {
                # 'top_level': 'main',
                'schema': metadata_base.CONTAINER_SCHEMA_VERSION,
                'structural_type': 'd3m.container.pandas.DataFrame',
                'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                'dimension': {
                    'name': 'rows',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularRow'],
                    'length': 4,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__'],
            'metadata': {
                'dimension': {
                    'name': 'columns',
                    'semantic_types': ['https://metadata.datadrivendiscovery.org/types/TabularColumn'],
                    'length': 3,
                },
            },
        }, {
            'selector': ['__ALL_ELEMENTS__', 0],
            'metadata': {'structural_type': 'numpy.int64', 'name': 'timestamp'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 1],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'values'},
        }, {
            'selector': ['__ALL_ELEMENTS__', 2],
            'metadata': {'structural_type': 'numpy.float64', 'name': 'b'},
        }])
        hyperparams_class = StatisticalMeanAbs.StatisticalMeanAbsPrimitive.metadata.get_hyperparams()

        hp = hyperparams_class.defaults().replace({
            'use_columns': [1,2],
            'use_semantic_types' : True,
            'window_size':2
        })

        primitive = StatisticalMeanAbs.StatisticalMeanAbsPrimitive(hyperparams=hp)

        output_main = primitive.produce(inputs=main).value
        print(output_main)
        expected_output = container.DataFrame(
            {'timestamp': [1, 3, 2, 5], 'values': [1.0, 2.0, 3.0, 4.0], 'b': [1.0, 4.0, 5.0, 6.0],
             'values_mean_abs': [1.5, 1.5, 2.5, 3.5], 'b_mean_abs': [2.5, 2.5, 4.5, 5.5]},
            columns=['timestamp', 'values', 'b', 'values_mean_abs', 'b_mean_abs'])

        self.assertEqual(output_main[['timestamp', 'values', 'b', 'values_mean_abs',
                                      'b_mean_abs']].values.tolist(), expected_output[
                             ['timestamp', 'values', 'b', 'values_mean_abs', 'b_mean_abs'
                              ]].values.tolist())

        self.assertEqual(utils.to_json_structure(output_main.metadata.to_internal_simple_structure()),
                         [{'metadata': {'dimension': {'length': 4,
                                                      'name': 'rows',
                                                      'semantic_types': [
                                                          'https://metadata.datadrivendiscovery.org/types/TabularRow']},
                                        'schema': 'https://metadata.datadrivendiscovery.org/schemas/v0/container.json',
                                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Table'],
                                        'structural_type': 'd3m.container.pandas.DataFrame'},
                           'selector': []},
                          {'metadata': {'dimension': {'length': 5,
                                                      'name': 'columns',
                                                      'semantic_types': [
                                                          'https://metadata.datadrivendiscovery.org/types/TabularColumn']}},
                           'selector': ['__ALL_ELEMENTS__']},
                          {'metadata': {'name': 'timestamp', 'structural_type': 'numpy.int64'},
                           'selector': ['__ALL_ELEMENTS__', 0]},
                          {'metadata': {'name': 'values', 'structural_type': 'numpy.float64'},
                           'selector': ['__ALL_ELEMENTS__', 1]},
                          {'metadata': {'name': 'b', 'structural_type': 'numpy.float64'},
                           'selector': ['__ALL_ELEMENTS__', 2]},
                          {'metadata': {'name': 'values_mean_abs',
                                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                                        'structural_type': 'numpy.float64'},
                           'selector': ['__ALL_ELEMENTS__', 3]},
                          {'metadata': {'name': 'b_mean_abs',
                                        'semantic_types': ['https://metadata.datadrivendiscovery.org/types/Attribute'],
                                        'structural_type': 'numpy.float64'},
                           'selector': ['__ALL_ELEMENTS__', 4]},

                          ])


        params = primitive.get_params()
        primitive.set_params(params=params)


if __name__ == '__main__':
    unittest.main()
