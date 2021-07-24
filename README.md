I write about the process of building this program in my blog at [troutstick.github.io](https://troutstick.github.io/);
please do check it out!

# ray_tracer
A parallelized CPU ray tracing program written in Rust. Makefile uses ImageMagick to convert output to PNG format.

Can render `.obj` files at arbitrary resolution. An example rendering by this program:

<img src="https://github.com/troutstick/ray_tracer/blob/master/images/example_output/teapot.png" width="400" height="400" />

This project was a super fun exercise in optimization, made all the more fun by its extensive use
of Rust. While ray tracing is already an embarassingly parallelizable problem,
using [Rayon](https://crates.io/crates/rayon) for parallel code made it even more unbelievably slick.

A primary challenge was writing large portions of the codebase [functionally](https://en.wikipedia.org/wiki/Functional_programming)
as opposed to [imperatively](https://en.wikipedia.org/wiki/Imperative_programming),
which is a very foreign style for programmers coming from C, or Java,
or pretty much any major language used in software development over the past several decades.
The reward was well worth it, since Rayon heavily relies on parallelism implemented over
Rust's iterators.

## Dependencies

In addition to dependencies handled by `cargo`, this program relies on ImageMagick for exporting `.png` files,
and on `eog` for displaying them. These additional tools are
not an essential part of the program unless you utilize the related commands defined in the Makefile.

This program was tested only for Linux machines.

## Running

Put any `.obj` files that you wish to render in `./images/input/`. Note that only a subset of `.obj` files
can be currently handled by this program.

Run the program using cargo.

`cargo r --release` generates 6 images of [a certain famous teapot](https://en.wikipedia.org/wiki/Utah_teapot).

`cargo r --release $OBJ_FILE_NAME` generates 6 images of a specified scene.

`cargo r --release $OBJ_FILE_NAME $NUM_ITER` generates `NUM_ITER` images.

Remember to run in release mode, or it will be excruciatingly slow. Happy rendering!
