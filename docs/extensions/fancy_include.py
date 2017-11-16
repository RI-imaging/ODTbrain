"""Include single scripts with doc string, code, and image

Use case
--------
There is an "examples" directory in the root of a repository,
e.g. 'include_doc_code_img_path = "../examples"' in conf.py
(default). An example is a file ("an_example.py") that consists
of a doc string at the beginning of the file, the example code,
and, optionally, an image file (png, jpg) ("an_example.png").


Configuration
-------------
In conf.py, set the parameter

   fancy_include_path = "../examples"

to wherever the included files reside.


Usage
-----
The directive

   .. fancy_include:: an_example.py

will display the doc string formatted with the first line as a
heading, a code block with line numbers, and the image file.
"""
import io
import os.path as op

from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
from docutils import nodes


class IncludeDirective(Directive):
    required_arguments = 1
    optional_arguments = 0

    def run(self):
        path = self.state.document.settings.env.config.fancy_include_path
        full_path = op.join(path, self.arguments[0])

        with io.open(full_path, "r") as myfile:
            text = myfile.read()

        # add reference
        name = op.basename(full_path)[:-3]
        rst = [".. _example_{}:".format(name),
               "",
               ]

        # add docstring
        source = text.split('"""')
        doc = source[1].split("\n")
        doc.insert(1, "~" * len(doc[0]))  # make title heading

        code = source[2].split("\n")

        for line in doc:
            rst.append(line)

        # image
        for ext in [".png", ".jpg"]:
            image_path = full_path[:-3] + ext
            if op.exists(image_path):
                break
        else:
            image_path = ""
        if image_path:
            rst.append(".. figure:: {}".format(image_path))
            rst.append("")

        # download file
        rst.append(":download:`{}<{}>`".format(
            op.basename(full_path), full_path))

        # code
        rst.append("")
        rst.append(".. code-block:: python")
        rst.append("   :linenos:")
        rst.append("")
        for line in code:
            rst.append("   {}".format(line))
        rst.append("")

        vl = ViewList(rst, "fakefile.rst")
        # Create a node.
        node = nodes.section()
        node.document = self.state.document
        # Parse the rst.
        nested_parse_with_titles(self.state, vl, node)
        return node.children


def setup(app):
    app.add_config_value('fancy_include_path', "../examples", 'html')

    app.add_directive('fancy_include', IncludeDirective)

    return {'version': '0.1'}   # identifies the version of our extension
