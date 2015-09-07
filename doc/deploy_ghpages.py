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
    travis encrypt GH_TOKEN="<token>" --add
    

4. Add the encrypted token to .travis.yml

    env:
      global:
      - GH_REF: github.com/<your name>/<your repo>.git
      - secure: "jdcn3kM/dI0zvVTn0UKgal8Br+745Qc1plaKXHcoKhwcwN+0Q1y5H1BnaF0KV2dWWeExVXMpqQMLOCylUSUmd30+hFqUgd3gFQ+oh9pF/+N72uzjnxHAyVjai5Lh7QnjN0SLCd2/xLYwaUIHjWbWsr5t2vK9UuyphZ6/F+7OHf+u8BErviE9HUunD7u4Q2XRaUF0oHuF8stoWbJgnQZtUZFr+qS1Gc3vF6/KBkMqjnq/DgBV61cWsnVUS1HVak/sGClPRXZMSGyz8d63zDxfA5NDO6AbPVgK02k+QV8KQCyIX7of8rBvBmWkBYGw5RnaeETLIAf6JrCKMiQzlJQZiMyLUvd/WflSIBKJyr5YmUKCjFkwvbKKvCU3WBUxFT2p7trKZip5JWg37OMvOAO8eiatf2FC1klNly1KHADU88QqNoi/0y2R/a+1Csrl8Gr/lXZkW4mMkI2due9epLwccDJtMF8Xc39EqRR46xA7Lx9vy7szYW5lLux3zwx1tH40wV6/dX4ZVFoWp/zfJw7TKdOHuOwjZuOuKp/shfJs94G9YCu7bBtvrGv9qCH2KiSgm1NJviwcsZWsVHaq1nP0LliDE7EM3Q0mnkYzlvfOOhA2G5Ka3rHl1RFj7+WYzO5GaAFWU7piP/kdBwc0Mu+hb6PMoy0oeLt39BDr29bNKMs="

5. Add the deploy command to .travis.yml

    after_success:
    - git config credential.helper "store --file=.git/credentials"
    - echo "https://${GH_TOKEN}:@github.com" > .git/credentials
    - if [[ $TRAVIS_PYTHON_VERSION == 3.4 ]]; then pip install numpydoc sphinx; fi
    - if [[ $TRAVIS_PYTHON_VERSION == 3.4 ]]; then python doc/deploy_ghpages.py; fi

"""
from __future__ import print_function
import os
from os.path import dirname, abspath
import subprocess as sp


# go to root of repository
os.chdir(dirname(dirname(abspath(__file__))))

# build sphinx
sp.check_output("python setup.py build_sphinx", shell=True)

# clone into new folder the gh-pages branch
sp.check_output("git config --global user.email 'travis@example.com'", shell=True)
sp.check_output("git config --global user.name 'Travis CI'", shell=True)
sp.check_output("git config --global credential.helper 'store --file=.git/credentials'", shell=True)
sp.check_output("echo 'https://${GH_TOKEN}:@github.com' > .git/credentials", shell=True)
sp.check_output("git clone --depth 1 -b gh-pages https://${GH_TOKEN}@${GH_REF} gh_pages", shell=True)

# copy everything from ./build/sphinx/html to ./gh_pages
#sp.check_output("cp -r ./build/sphinx/html/* ./gh_pages/", shell=True)
sp.check_output("rsync -rt --del --exclude='.git' --exclude='.nojekyll' ./build/sphinx/html/* ./gh_pages/", shell=True)

# commit changes
os.chdir("gh_pages")
sp.check_output("echo 'https://${GH_TOKEN}:@github.com' > .git/credentials", shell=True)
sp.check_output("git add --all ./*", shell=True)

try:
    # If there is nothing to commit, then 'git commit' returns non-zero exit status
    errorcode = sp.check_output("git commit -a -m 'travis bot build {} [ci skip]'".format(os.getenv("TRAVIS_COMMIT")), shell=True)
    print("git commit returned:", errorcode)
except:
    pass
else:
    sp.check_output("git push --force --quiet origin gh-pages", shell=True)

