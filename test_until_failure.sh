COUNT=-1

trap ctrl_c INT
function ctrl_c() {
    MAX_COUNT=$COUNT
}

while [ $? -eq "0" ]; do
    ((COUNT = $COUNT + 1))
    if [ -n "${MAX_COUNT}" ]; then
        if [ "${COUNT}" -ge "${MAX_COUNT}" ]; then
            break
        fi
    fi
    echo "Running tests..."
    python setup.py nosetests
done

echo "Ran tests $COUNT time(s)"
