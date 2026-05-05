export default function Hero({ onTrySimulation, onHowItWorks }) {
  return (
    <section className="relative overflow-hidden pt-20 pb-24">

      <div className="absolute inset-0 bg-hero-gradient opacity-60"></div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 md:px-10">
        <div className="bg-white/70 backdrop-blur-xl rounded-3xl p-10 md:p-14 shadow-md">

          <div className="flex flex-col md:flex-row items-center gap-10">
            <div className="flex-1">
              <h1 className="text-5xl md:text-6xl font-extrabold leading-tight text-gray-900">
                Transform Your
                <span className="bg-accent-gradient bg-clip-text text-transparent"> Future Look </span>
                with
                <span className="bg-accent-gradient bg-clip-text text-transparent"> AI Precision</span>
              </h1>

              <p className="text-gray-600 mt-5 max-w-xl text-lg">
                Experience ultra-realistic AI simulations for face enhancements —
                Botox, fillers, jawline refinement, skin treatments & more.
              </p>

              <div className="mt-8 flex flex-wrap gap-4">
                <button
                  onClick={onTrySimulation}
                  className="px-6 py-3 rounded-xl bg-accent-gradient text-white font-semibold shadow-md hover:shadow-lg hover:scale-105 hover:-translate-y-0.5 transition-all duration-300"
                >
                  Try Simulation
                </button>

                <button
                  onClick={onHowItWorks}
                  className="px-6 py-3 rounded-xl border border-brandPurple text-brandPurple font-semibold hover:bg-brandPurple hover:text-white hover:scale-105 transition-all duration-300"
                >
                  How It Works
                </button>
              </div>
            </div>

            <div className="flex-1 flex justify-center">
              <div className="w-72 h-72 rounded-3xl bg-accent-gradient shadow-md"></div>
            </div>

          </div>
        </div>
      </div>
    </section>
  );
}
