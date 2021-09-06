use parallel_playground::*;

fn main() {
    let input = &[394848, 33, 3, 2, 2, 2, 32, 32, 3];

    let context = Context::new();

    let buffer = context.upload(input);

    context.do_in_pass(|pass| {
        pass.mod_buffer_in_place(&buffer, 9);

        let indices = pass.upload(&[1, 3]);

        pass.scatter_with_value(&indices, &buffer, 77);
    });

    let read = context.read_buffer(&buffer);

    println!("{:?}", &*read);
}
