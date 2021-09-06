use wgpu::util::DeviceExt;

pub struct Context {
    device: wgpu::Device,
    queue: wgpu::Queue,
    mod_pipeline: wgpu::ComputePipeline,
    readwrite_bind_group_layout: wgpu::BindGroupLayout,
    io_bind_group_layout: wgpu::BindGroupLayout,
    scatter_with_value_pipeline: wgpu::ComputePipeline,
    sum_pipeline: wgpu::ComputePipeline,
}

impl Context {
    pub fn new() -> Self {
        env_logger::init();

        let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .unwrap();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("device"),
                features: wgpu::Features::SPIRV_SHADER_PASSTHROUGH | wgpu::Features::PUSH_CONSTANTS,
                limits: wgpu::Limits {
                    max_push_constant_size: std::mem::size_of::<u32>() as u32,
                    ..Default::default()
                },
            },
            None,
        ))
        .unwrap();

        let io_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("IO bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let readwrite_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("readwrite bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let io_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("IO pipeline layout"),
            bind_group_layouts: &[&io_bind_group_layout],
            push_constant_ranges: &[],
        });

        let io_push_constant_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("IO push constant pipeline layout"),
                bind_group_layouts: &[&io_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<u32>() as u32,
                }],
            });

        let readwrite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("readwrite pipeline layout"),
                bind_group_layouts: &[&readwrite_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..std::mem::size_of::<u32>() as u32,
                }],
            });

        let sum_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sum"),
            layout: Some(&io_pipeline_layout),
            module: &unsafe {
                device.create_shader_module_spirv(&wgpu::include_spirv_raw!("sum.comp.spv"))
            },
            entry_point: "main",
        });

        let scatter_with_value_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("scatter_with_value"),
                layout: Some(&io_push_constant_pipeline_layout),
                module: &unsafe {
                    device.create_shader_module_spirv(&wgpu::include_spirv_raw!(
                        "scatter_with_value.comp.spv"
                    ))
                },
                entry_point: "main",
            });

        let mod_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("mod"),
            layout: Some(&readwrite_pipeline_layout),
            module: &unsafe {
                device.create_shader_module_spirv(&wgpu::include_spirv_raw!("mod.comp.spv"))
            },
            entry_point: "main",
        });

        Self {
            device,
            queue,
            mod_pipeline,
            readwrite_bind_group_layout,
            io_bind_group_layout,
            scatter_with_value_pipeline,
            sum_pipeline,
        }
    }

    pub fn upload<T: bytemuck::Pod>(&self, bytes: &[T]) -> MappableStorageBuffer<T> {
        MappableStorageBuffer {
            inner: self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(bytes),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::MAP_READ,
                }),
            size: bytes.len() as u32,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn do_in_pass<T: Fn(&mut Pass)>(&self, closure: T) {
        let mut pass = Pass {
            context: self,
            command_encoder: self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }),
        };

        closure(&mut pass);

        self.queue
            .submit(std::iter::once(pass.command_encoder.finish()));
    }

    pub fn read_buffer<'a, T>(&self, buffer: &'a MappableStorageBuffer<T>) -> MappedBuffer<'a, T> {
        let slice = buffer.inner.slice(..);

        let map_future = slice.map_async(wgpu::MapMode::Read);

        self.device.poll(wgpu::Maintain::Wait);

        pollster::block_on(map_future).unwrap();

        MappedBuffer {
            view: slice.get_mapped_range(),
            _phantom: buffer._phantom,
        }
    }
}

pub struct Pass<'a> {
    context: &'a Context,
    command_encoder: wgpu::CommandEncoder,
}

impl<'a> Pass<'a> {
    pub fn mod_buffer_in_place(&mut self, buffer: &MappableStorageBuffer<u32>, value: u32) {
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.context.readwrite_bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.inner.as_entire_binding(),
                }],
            });

        let mut compute_pass = self
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

        compute_pass.set_pipeline(&self.context.mod_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.set_push_constants(0, bytemuck::bytes_of(&value));
        compute_pass.dispatch(dispatch_count(buffer.size, 64), 1, 1);
    }

    pub fn scatter_with_value(
        &mut self,
        indices: &MappableStorageBuffer<u32>,
        output: &MappableStorageBuffer<u32>,
        value: u32,
    ) {
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.context.io_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: indices.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.inner.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = self
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

        compute_pass.set_pipeline(&self.context.scatter_with_value_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.set_push_constants(0, bytemuck::bytes_of(&value));
        compute_pass.dispatch(dispatch_count(indices.size, 64), 1, 1);
    }

    pub fn sum(
        &mut self,
        inputs: &MappableStorageBuffer<u32>,
        output: &MappableStorageBuffer<u32>,
    ) {
        let bind_group = self
            .context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.context.io_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: inputs.inner.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output.inner.as_entire_binding(),
                    },
                ],
            });

        let mut compute_pass = self
            .command_encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });

        compute_pass.set_pipeline(&self.context.sum_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch(dispatch_count(inputs.size, 64), 1, 1);
    }

    pub fn upload<T: bytemuck::Pod>(&self, bytes: &[T]) -> MappableStorageBuffer<T> {
        self.context.upload(bytes)
    }
}

pub struct MappedBuffer<'a, T> {
    view: wgpu::BufferView<'a>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T: bytemuck::Pod> std::ops::Deref for MappedBuffer<'a, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        bytemuck::cast_slice(&self.view)
    }
}

pub struct MappableStorageBuffer<T> {
    inner: wgpu::Buffer,
    size: u32,
    _phantom: std::marker::PhantomData<T>,
}

fn dispatch_count(num: u32, group_size: u32) -> u32 {
    let mut count = num / group_size;
    let rem = num % group_size;
    if rem != 0 {
        count += 1;
    }

    count
}
