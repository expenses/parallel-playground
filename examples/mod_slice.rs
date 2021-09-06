use parallel_playground::*;

fn main() {
    let context = Context::new();

    let buffer = context.upload(&[2; 100024]);

    let output = context.upload(&[0]);

    context.do_in_pass(|pass| {
        //pass.mod_buffer_in_place(&buffer, 9);

        //let indices = pass.upload(&[1, 3]);

        //pass.scatter_with_value(&indices, &buffer, 77);

        pass.sum(&buffer, &output);
    });

    let read = context.read_buffer(&buffer);

    let x = context.read_buffer(&output);

    println!("{:?}", &*x);
}
