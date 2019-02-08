"""Display changelog with links to GitHub issues

Usage
-----
The directive

   .. include_changelog:: ../CHANGELOG

adds the content of the changelog file into the current document.
References to GitHub issues are identified as "(#XY)" (with parentheses
and hash) and a link is inserted

 https://github.com/RI-imaging/{PROJECT}/issues/{XY}

where PROJECT ist the `project` variable defined in conf.py.
"""
import io
import re

from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
from docutils import nodes


class IncludeDirective(Directive):
    required_arguments = 1
    optional_arguments = 0

    def run(self):
        full_path = self.arguments[0]
        project = self.state.document.settings.env.config.github_project

        def insert_github_link(reobj):
            line = reobj.string
            instr = line[reobj.start():reobj.end()]
            issue = instr.strip("#()")
            link = "https://github.com/{}/issues/".format(project)
            rstlink = "(`#{issue} <{link}{issue}>`_)".format(issue=issue,
                                                             link=link)
            return rstlink

        with io.open(full_path, "r") as myfile:
            text = myfile.readlines()

        rst = []
        for line in text:
            line = line.strip("\n")
            if line.startswith("  ") and line.strip().startswith("-"):
                # list in list:
                rst.append("")
            if not line.startswith(" "):
                rst.append("")
                line = "version " + line
                rst.append(line)
                rst.append("-"*len(line))
            elif not line.strip():
                rst.append(line)
            else:
                line = re.sub(r"\(#[0-9]*\)", insert_github_link, line)
                rst.append(line)

        vl = ViewList(rst, "fakefile.rst")
        # Create a node.
        node = nodes.section()
        node.document = self.state.document
        # Parse the rst.
        nested_parse_with_titles(self.state, vl, node)
        return node.children


def setup(app):
    app.add_config_value('github_project', "user/project", 'html')
    app.add_directive('include_changelog', IncludeDirective)
    return {'version': '0.1'}   # identifies the version of our extension
