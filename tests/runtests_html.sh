#!/bin/bash
# Go to script directory
cd $(dirname $0)

OUTFILE='pytest.html'


PYTHON="python"
PYTHON2_7="python2.7"
PYTHON3_4="python3.4"

PYLIST=( $PYTHON )

CLEAN="yes"

while getopts ":p:ahn" opt; do
    case $opt in
        a)
            echo "run all!" >&2
            PYLIST=( $PYTHON2_7 $PYTHON3_4 )
            ;;
        p)
            if [ "$OPTARG" = "2.7" ]; then
                echo "run with python2.7"
                PYLIST=( $PYTHON2_7 )
            elif [ "$OPTARG" = "3.4" ]; then
                echo "run with python3.4"
                PYLIST=( $PYTHON3_4 )
            else
                echo "arguent $OPTARG for -p not unterstood"
                exit 1
            fi
            ;;
        n)
            CLEAN="no"
            ;;
        h)
            echo "run PYTEST included in runtests.py"
            echo "and write the output in html format to '$OUTFILE'"
            echo "to choose the python version use the following switches"
            echo ""
            echo "    -p VER"
            echo "        VER = 2.7 -> uses $PYTHON2_7"
            echo "        VER = 3.4 -> uses $PYTHON3_4"
            echo ""
            echo "    -a"
            echo "        runs both, $PYTHON2_7 and $PYTHON3_4"
            echo ""
            echo "if neither -p nor -a are given, the default interpreter '$PYTHON' is used"
            echo ""
            echo "    -n"
            echo "        no cleanup after all"
            echo ""
            exit 0
            ;;
            
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "option $OPTARG requires an argument"
            exit 1
            ;;
    esac
done

echo "Working directory: $(pwd)"
rm -f -v $OUTFILE
touch $OUTFILE
for py in ${PYLIST[@]}; do
    $py --version
    echo "Running py.test ..."
    (echo ""; date; $py runtests.py --color=yes) | tee -a $OUTFILE
    echo "Done!"
done

cat $OUTFILE | aha --black --title "pytest output for jobmanager module" > $OUTFILE

if [ "$CLEAN" = "yes" ]; then
    rm -f *.trb
    rm -f *.dump
    rm -f *.db
fi

echo "ALL DONE! (output written to $OUTFILE)"
