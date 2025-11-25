class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.generator_a_to_b = Generator()
        self.generator_b_to_a = Generator()
        self.discriminator_a = Discriminator()
        self.discriminator_b = Discriminator()

    def forward(self, real_a, real_b):
        fake_b = self.generator_a_to_b(real_a)
        cycle_a = self.generator_b_to_a(fake_b)

        fake_a = self.generator_b_to_a(real_b)
        cycle_b = self.generator_a_to_b(fake_a)

        return cycle_a, cycle_b
