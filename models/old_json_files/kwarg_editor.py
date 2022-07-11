import json
import re
from functools import partial
from typing import Callable, Dict
# noinspection PyUnresolvedReferences
from inspect import signature, _empty

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash.dependencies import Input, ALL, Output, State

from targeted_callbacks import targeted_callback

"""
{'name', 'title', 'value', 'type', 
"""


def regularize_name(name):
    return ''.join([c for c in name if c not in [' ']])


class SimpleItem(dbc.FormGroup):
    def __init__(self,
                 name,
                 base_id,
                 title=None,
                 type='number',
                 debounce=True,
                 visible=True,
                 **kwargs):
        self.name = name

        self.label = dbc.Label(title or name)
        self.input = dbc.Input(type=type,
                               debounce=debounce,
                               id={**base_id,
                                   'name': name,
                                   'layer': 'input'},
                               **kwargs)
        style = {}
        if not visible:
            style['display'] = 'none'

        super(SimpleItem, self).__init__(id={**base_id,
                                             'name': name,
                                             'layer': 'form_group'},
                                         children=[self.label, self.input],
                                         style=style)


class FloatItem(SimpleItem):
    pass


class IntItem(SimpleItem):
    def __init__(self, *args, **kwargs):
        if 'min' not in kwargs:
            kwargs['min'] = -9007199254740991  # min must be set for int validation to be enabled
        super(IntItem, self).__init__(*args, step=1, **kwargs)


class StrItem(SimpleItem):
    def __init__(self, *args, **kwargs):
        super(StrItem, self).__init__(*args, type='text', **kwargs)


class SliderItem(dbc.FormGroup):
    def __init__(self,
                 name,
                 base_id,
                 title=None,
                 visible=True,
                 **kwargs):

        self.label = dbc.Label(title or name)
        self.input = dcc.Slider(id={**base_id,
                                    'name': name,
                                    'layer': 'input'},
                                **kwargs)

        style = {}
        if not visible:
            style['display'] = 'none'

        super(SliderItem, self).__init__(id={**base_id,
                                             'name': name,
                                             'layer': 'form_group'},
                                         children=[self.label, self.input],
                                         style=style)


class ChecklistItem(dbc.FormGroup):
    def __init__(self,
                 name,
                 base_id,
                 options,
                 title=None,
                 visible=True,
                 **kwargs):

        self.label = dbc.Label(title or name)
        self.input = dcc.Checklist(id={**base_id,
                                       'name': name,
                                       'layer': 'input'},
                                   options=options,
                                   **kwargs)

        style = {}
        if not visible:
            style['display'] = 'none'

        super(ChecklistItem, self).__init__(id={**base_id,
                                                'name': name,
                                                'layer': 'form_group'},
                                            children=[self.label, self.input],
                                            style=style)


class DropdownItem(dbc.FormGroup):
    def __init__(self,
                 name,
                 base_id,
                 options,
                 title=None,
                 visible=True,
                 **kwargs):

        self.label = dbc.Label(title or name)
        self.input = dcc.Dropdown(id={**base_id,
                                      'name': name,
                                      'layer': 'input'},
                                  options=options,
                                  **kwargs)

        style = {}
        if not visible:
            style['display'] = 'none'

        super(DropdownItem, self).__init__(id={**base_id,
                                               'name': name,
                                               'layer': 'form_group'},
                                           children=[self.label, self.input],
                                           style=style)


class RadioItem(dbc.FormGroup):
    def __init__(self,
                 name,
                 base_id,
                 options,
                 title=None,
                 visible=True,
                 **kwargs):

        self.label = dbc.Label(title or name)
        self.input = dbc.RadioItems(id={**base_id,
                                        'name': name,
                                        'layer': 'input'},
                                    options=options,
                                    **kwargs)

        style = {}
        if not visible:
            style['display'] = 'none'

        super(RadioItem, self).__init__(id={**base_id,
                                            'name': name,
                                            'layer': 'form_group'},
                                        children=[self.label, self.input],
                                        style=style)


class BoolItem(dbc.FormGroup):
    def __init__(self,
                 name,
                 base_id,
                 title=None,
                 visible=True,
                 **kwargs):

        self.label = dbc.Label(title or name)
        self.input = daq.ToggleSwitch(id={**base_id,
                                          'name': name,
                                          'layer': 'input'},
                                      **kwargs)
        self.output_label = dbc.Label('False/True')

        style = {}
        if not visible:
            style['display'] = 'none'

        super(BoolItem, self).__init__(id={**base_id,
                                           'name': name,
                                           'layer': 'form_group'},
                                       children=[self.label, self.input, self.output_label],
                                       style=style)


class ParameterEditor(dbc.Form):

    type_map = {float: FloatItem,
                int: IntItem,
                str: StrItem,
                }

    def __init__(self, _id, parameters, **kwargs):
        self._parameters = parameters

        super(ParameterEditor, self).__init__(id=_id, children=[], className='kwarg-editor', **kwargs)
        self.children = self.build_children()

    def init_callbacks(self, app):
        targeted_callback(self.stash_value,
                          Input({**self.id,
                                 'name': ALL,
                                 'layer': 'input'},
                                'value'),
                          Output(self.id, 'n_submit'),
                          State(self.id, 'n_submit'),
                          app=app)

    def stash_value(self, value):
        # find the changed item name from regex
        r = '(?<=\"name\"\:\")[\w\-_]+(?=\")'
        matches = re.findall(r, dash.callback_context.triggered[0]['prop_id'])

        if not matches:
            raise LookupError('Could not find changed item name. Check that all parameter names use simple chars (\\w)')

        name = matches[0]
        self.parameters[name]['value'] = value

        print(self.values)

        return next(iter(dash.callback_context.states.values())) or 0 + 1

    @property
    def values(self):
        return {param['name']: param.get('value', None) for param in self._parameters}

    @property
    def parameters(self):
        return {param['name']: param for param in self._parameters}

    def _determine_type(self, parameter_dict):
        if 'type' in parameter_dict:
            if parameter_dict['type'] in self.type_map:
                return parameter_dict['type']
            elif parameter_dict['type'].__name__ in self.type_map:
                return parameter_dict['type'].__name__
        elif type(parameter_dict['value']) in self.type_map:
            return type(parameter_dict['value'])
        raise TypeError(f'No item type could be determined for this parameter: {parameter_dict}')

    def build_children(self, values=None):
        children = []
        for parameter_dict in self._parameters:
            parameter_dict = parameter_dict.copy()
            if values and parameter_dict['name'] in values:
                parameter_dict['value'] = values[parameter_dict['name']]
            type = self._determine_type(parameter_dict)
            parameter_dict.pop('type', None)
            item = self.type_map[type](**parameter_dict, base_id=self.id)
            children.append(item)

        return children


class JSONParameterEditor(ParameterEditor):
    type_map = {'float': FloatItem,
                'int': IntItem,
                'str': StrItem,
                'intslider': SliderItem,
                'strdropdown': DropdownItem,
                'radio': RadioItem,
                'bool': BoolItem,
                "strchecklist": ChecklistItem
                }

    def _determine_type(self, parameter_dict):
        if 'type' in parameter_dict:
            if parameter_dict['type'] in self.type_map:
                return parameter_dict['type']
            elif parameter_dict['type'].__name__ in self.type_map:
                return parameter_dict['type'].__name__
        elif type(parameter_dict['value']).__name__ in self.type_map:
            return type(parameter_dict['value']).__name__
        raise TypeError(f'No item type could be determined for this parameter: {parameter_dict}')


class KwargsEditor(ParameterEditor):
    def __init__(self, instance_index, func: Callable, **kwargs):
        self.func = func
        self._instance_index = instance_index

        parameters = [{'name': name, 'value': param.default} for name, param in
                      signature(func).parameters.items()
                      if param.default is not _empty]

        super(KwargsEditor, self).__init__(dict(index=instance_index, type='kwargs-editor'),
                                           parameters=parameters, **kwargs)

    @staticmethod
    def parameters_from_func(func, prefix=''):
        parameters = [{'name': prefix + name,
                       'title': name,
                       'value': param.default}
                      for name, param in signature(func).parameters.items()
                      if param.default is not _empty]
        return parameters

    def new_record(self):
        return {name: p.default for name, p in signature(self.func).parameters.items() if p.default is not _empty}


class StackedKwargsEditor(html.Div):
    def __init__(self, instance_index, funcs: Dict[str, Callable], selector_label: str, id='kwargs-editor', **kwargs):
        self.func_selector = dbc.Select(id=dict(index=instance_index, type=id, layer='stack'),
                                        options=[{'label': name, 'value': name} for i, name in enumerate(funcs.keys())],
                                        value=next(iter(funcs.keys())))

        self.funcs = funcs

        parameters = []
        for i, (name, func) in enumerate(funcs.items()):
            regularized_name = regularize_name(name)
            func_params = KwargsEditor.parameters_from_func(func, prefix=f'{regularized_name}-')
            if i:
                for param in func_params:
                    param['visible'] = False
            # self._param_map[name] = func_params.keys())
            parameters.extend(func_params)

        self.parameter_editor = ParameterEditor(dict(index=instance_index, type=id, layer='editor'),
                                                parameters=parameters,
                                                **kwargs)

        super(StackedKwargsEditor, self).__init__(children=[dbc.Label(selector_label),
                                                            html.Br(),
                                                            self.func_selector,
                                                            dbc.CardBody(children=self.parameter_editor)])

    def init_callbacks(self, app):
        for child in self.parameter_editor.children:
            targeted_callback(partial(self.update_visibility, name=child.id['name']),
                              Input(self.func_selector.id, 'value'),
                              Output(child.id, 'style'),
                              prevent_initial_call=True,
                              app=app)
        self.parameter_editor.init_callbacks(app)

    def update_visibility(self, value: str, name:str):
        if name.startswith(f'{regularize_name(value)}-'):
            return {'display': 'block'}
        else:
            return {'display': 'none'}


if __name__ == '__main__':

    app_kwargs = {'external_stylesheets': [dbc.themes.BOOTSTRAP]}
    app = dash.Dash(__name__, **app_kwargs)

    item_list = ParameterEditor(_id={'type': 'parameter_editor'},
                                parameters=[{'name': 'test', 'value': 2, 'max': 10, 'min': 0},
                                            {'name': 'test2', 'value': 'blah'},
                                            {'name': 'test3', 'value': 3.2, 'type': float}])

    with open('example.json') as f:
        json_file = json.load(f)

    json_items = JSONParameterEditor(_id={'type': 'json_parameter_editor'},
                                     parameters=json_file)

    def my_func(a='t', p=1, d='blah', e=23.4):
        ...

    def my_func2(a, b, x=1, w='blah', z=23.4):
        ...

    kwarg_list = KwargsEditor(0, func=my_func)

    func_editor = StackedKwargsEditor(1, funcs={'my_func': my_func, 'my_func2': my_func2},
                                      selector_label='test')

    kwarg_list.init_callbacks(app)
    item_list.init_callbacks(app)
    func_editor.init_callbacks(app)
    json_items.init_callbacks(app)

    app.layout = html.Div([json_items])

    app.run_server(debug=True)
