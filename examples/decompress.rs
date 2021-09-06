use parallel_playground::*;

fn main() {
    env_logger::init();

    let input = [5, 4, 0, 3, 5, 5, 5, 5, 5, 5];

    let context = Context::new();

    let buffer = context.upload(&input[..]);

    let size = {
        let output = context.upload(&0);

        context.do_in_pass(|pass| {
            pass.sum(&buffer, &output);
        });

        let output = context.read_buffer(&output);

        *output
    };

    let output = context.storage_buffer_of_length::<u32>(size);

    println!("{:?}", &*context.read_buffer(&output));

    /*let scanned_input = exclusive_scan(&input);

    let output_size = sum(&input);

    let mut output = zeroed_array_of_size(output_size);

    println!("{:?}", scanned_input);

    scatter_with_value(&mut output, &scanned_input, 1);

    println!("{:?}", output);

    let x = inclusive_scan(&output);

    println!("{:?}", x);

    let mapped = x.iter().map(|&v| v % 2).collect::<Vec<_>>();

    println!("{:?}", mapped);*/
}
