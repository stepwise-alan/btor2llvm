#!/usr/bin/env sh

TIMEOUT=10
tmp_dir=/tmp

filename=$(basename -- "$1")
test_name=${filename%.*}
test_dir=$tmp_dir/$test_name
out_path=$test_dir/a.out

[ -d "$tmp_dir" ] || mkdir "$tmp_dir"
[ -d "$test_dir" ] || mkdir "$test_dir"

#clang -fsanitize=fuzzer "$1" -flto -fuse-ld=lld -Wl,-plugin-opt,save-temps
clang -fsanitize=fuzzer "$1" -flto -fuse-ld=lld -Wl -o "$out_path"

cd "$test_dir" || exit
"$out_path" -max_total_time=$TIMEOUT

exit_status=$?
if [ $exit_status -eq 77 ]; then
    echo sat
else
    echo unknown
fi

rm -rf "$test_dir"
