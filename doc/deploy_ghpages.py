#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publish the documentation on GitHub Pages.

Prerequisites
-------------

1. Create empty gh-pages branch:

    git branch gh-pages
    git checkout gh-pages
    git symbolic-ref HEAD refs/heads/gh-pages
    rm .git/index
    git clean -fdx


2. Setup sphinx.

    python setup.py build_sphinx
   
   should create a build/sphinx/html folder in the repository root. 


3. Create GitHub repo token and encrypt it 

    gem install travis
    travis encrypt -r <username>/<repo> GIT_NAME="Your Name" GIT_EMAIL=you@example.com GH_TOKEN=<token>

4. Add the encrypted token to .travis.yml

    env:
      global:
      - secure: "PUjnq3yn84H9fxhceq9JmykWywz9pCoy4j/+GruTQsO1OCuZEcvvMWDmqEzSDzOLL7rb5hE9GXV9YJaC9yiTpl6nhRojL710lEK+9jK+cv49EA/lvWaxSzicdCiEDkDjbhKmrTZtfprCLKIXC2C843vFSOspLuyMiMBquKBsecY43lw+Z9ZQePxbhNzeIDp8wbsW+gIAteI7BjT/fZ+KroGdkyyQMfHTakxNZCJEK38RwR3OuAaZ931P3XXuh7fnEEIHcTbvd7gSfiokmkGL/6vMBTids3uzOnv+WAH6zSiy9+fAPnZqOe06/+1sAg2Iu0/GFLHCXaHREpRyFb7CBQ2tJ5ZnvETahMsbA3cZc67vhs1Zg56PMhDpB0Hvg3vksV4FipA3xJgExsxVvAahvEiq3D6rs6yDfrNgbRi+Ha/zQ9EP/eNWqWsVnDMG0yce6t6xbCXBj9/J5UBYnHiY6E54V9SM3/9mCnM79Z2qOCDZEoJzGENNDzpggeNa0F5ihdK2AZ5HzncX+H69jZM4azFlqKxdW3d9RRafkggnJ/VMInHciJKu+4VuNWRn8IHKCS2q55n4gJtgeJdUkzjjAv4zSNpfX70X66O26asqJWsrGMXyYyPyVxmD3PSx7ZsdHoU2OnqzH8nSv+tlIM0fsW8551Zka+dmsmC3UzG88RE="

"""
import os
from os.path import dirname, abspath
import subprocess as sp


# go to root of repository
os.chdir(dirname(dirname(abspath(__file__))))

# build sphinx
sp.check_output("python setup.py build_sphinx", shell=True)

# clone into new folder the gh-pages branch
sp.check_output("git clone --depth 1 -b gh-pages https://github.com/paulmueller/ODTbrain.git gh_pages", shell=True)

# copy everything from ./build/sphinx/html to ./gh_pages
sp.check_output("cp -r ./build/sphinx/html/* ./gh_pages/", shell=True)

# commit changes
os.chdir("gh_pages")
sp.check_output("git add ./*", shell=True)
sp.check_output("git commit -a -m 'travis bot doc build'", shell=True)
sp.check_output("git push", shell=True)

