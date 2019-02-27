set -e

while [ $? -eq "0" ]; do
    echo "Running tests..."
    python setup.py nosetests
done
